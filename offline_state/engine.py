"""
Simulation engine for OfflineState distributed cache.

Tick-based simulation where each tick = 1 base RDMA RTT (~1us).
Clients have local LRU caches + bloom filters. Reads check:
  1. Local cache (0 ticks)
  2. Peer broadcast via bloom filter (1 tick, same rack)
  3. Server via one-sided RDMA read (server_rtt_ticks, cross-rack)

Updates use one-sided RDMA write to server (server_rtt_ticks).
An offline sync worker periodically rebuilds and merges bloom filters.
"""

import time
from collections import defaultdict

import numpy as np

from offline_state.config import OfflineStateConfig
from offline_state.bloom_filter import BloomFilter
from offline_state.gibf import GroupIndexedBloomFilter
from offline_state.client import (
    OfflineClient,
    PHASE_IDLE, PHASE_PEER_READ, PHASE_PEER_READ_GROUP,
    PHASE_SERVER_READ, PHASE_SERVER_WRITE,
)
from sim_config import OPS_PER_CLIENT, MIN_OPS


def _any_peer_has_key(clients, requesting_client_id, key):
    """Check if any peer client has the key in its cache (simulated broadcast)."""
    for c in clients:
        if c.client_id == requesting_client_id:
            continue
        # Exact cache lookup (simulation uses ground truth, not bloom filter)
        if key in c.cache:
            return True
    return False


def _any_group_peer_has_key(clients_by_group, requesting_client_id, key, target_groups):
    """Check if any peer in the target groups has the key (GIBF targeted query)."""
    for gid in target_groups:
        for c in clients_by_group[gid]:
            if c.client_id != requesting_client_id and key in c.cache:
                return True
    return False


def run_offline_sync(clients, gibf=None):
    """
    Offline sync worker: rebuild bloom filters and GIBF.

    1. Each client rebuilds its local bloom filter from current cache keys
    2. Build GIBF: for each client's cached keys, set group bit at hash positions
    3. Share GIBF reference with all clients
    """
    for c in clients:
        c.local_bloom.rebuild_from_keys(c.cache.keys())

    if gibf is not None:
        gibf.clear()
        for c in clients:
            gid = c.group_id
            for key in c.cache:
                gibf.add(key, gid)
        for c in clients:
            c.gibf = gibf
    else:
        # Fallback: legacy merged bloom
        local_blooms = [c.local_bloom for c in clients]
        merged = BloomFilter.bitwise_or(local_blooms)
        for c in clients:
            c.peer_bloom = merged.copy()


def tick_client(client, clients, config, workload_keys, workload_is_read,
                op_idx_holder, current_tick=0, clients_by_group=None):
    """
    Advance one client by one tick.
    Local cache is free, 0 RTT, the client loops and processes
    multiple local hits within a single tick, simulates that local memory access
    (~100ns) is ~30x faster than an RDMA RTT (~3us).

    Returns True if at least one operation completed this tick.
    """
    completed_any = False

    # Loop: local cache hits are free, so keep going until we hit
    # a non-local path or a non-IDLE phase.
    while client.phase == PHASE_IDLE:
        idx = op_idx_holder[0]
        if idx >= len(workload_keys):
            idx = 0
        op_idx_holder[0] = idx + 1

        key = int(workload_keys[idx])
        client.op_key = key
        is_read = workload_is_read[idx]
        client.op_start_tick = current_tick

        if is_read:
            client.op_type = 'read'
            # Check local cache first (0 RTT, 0 RDMA calls)
            if client.cache_get(key) is not None:
                client.ops_completed += 1
                client.local_hits += 1
                client.read_latencies.append(0)
                completed_any = True
                continue  # FREE — try next op in same tick

            # Check GIBF or peer bloom filter
            if client.gibf is not None:
                candidate_groups = client.gibf.query_excluding(
                    key, client.group_id)
                if candidate_groups:
                    client.target_groups = candidate_groups
                    client.phase = PHASE_PEER_READ_GROUP
                    break
            elif client.peer_bloom.contains(key):
                client.phase = PHASE_PEER_READ
                break

            # Go to server
            client.phase = PHASE_SERVER_READ
            break
        else:
            client.op_type = 'update'
            client.phase = PHASE_SERVER_WRITE
            break

    if client.phase == PHASE_PEER_READ:
        # Broadcast to peers (1 tick, same rack) — N-1 actual RDMA reads
        client.rdma_calls += len(clients) - 1
        if _any_peer_has_key(clients, client.client_id, client.op_key):
            client.cache_put(client.op_key, client.op_key)
            client.ops_completed += 1
            client.peer_hits += 1
            client.read_latencies.append(current_tick - client.op_start_tick + 1)
            client.phase = PHASE_IDLE
            return True
        else:
            # False positive — fall through to server next tick
            client.false_positives += 1
            client.phase = PHASE_SERVER_READ
            return False

    if client.phase == PHASE_PEER_READ_GROUP:
        # GIBF: targeted query to candidate groups only (1 tick, same rack)
        num_groups = len(client.target_groups)
        group_size = len(clients) // client.gibf.num_groups if client.gibf else 1
        client.rdma_calls += num_groups * group_size
        client.group_peer_queries += 1
        if _any_group_peer_has_key(clients_by_group, client.client_id,
                                   client.op_key, client.target_groups):
            client.cache_put(client.op_key, client.op_key)
            client.ops_completed += 1
            client.peer_hits += 1
            client.read_latencies.append(current_tick - client.op_start_tick + 1)
            client.phase = PHASE_IDLE
            return True
        else:
            # False positive — fall through to server next tick
            client.false_positives += 1
            client.phase = PHASE_SERVER_READ
            return False

    if client.phase == PHASE_SERVER_READ:
        # One-sided RDMA read from server (server_rtt_ticks ticks)
        if client.ticks_remaining == 0:
            client.ticks_remaining = config.server_rtt_ticks
            client.rdma_calls += 1
        client.ticks_remaining -= 1
        if client.ticks_remaining > 0:
            return False
        client.cache_put(client.op_key, client.op_key)
        client.ops_completed += 1
        client.server_hits += 1
        client.read_latencies.append(current_tick - client.op_start_tick + 1)
        client.phase = PHASE_IDLE
        return True

    if client.phase == PHASE_SERVER_WRITE:
        # One-sided RDMA write to server (server_rtt_ticks ticks)
        if client.ticks_remaining == 0:
            client.ticks_remaining = config.server_rtt_ticks
            client.rdma_calls += 1
        client.ticks_remaining -= 1
        if client.ticks_remaining > 0:
            return False
        if client.op_key in client.cache:
            client.cache_put(client.op_key, client.op_key + 1)
        client.ops_completed += 1
        client.write_latencies.append(current_tick - client.op_start_tick + 1)
        client.phase = PHASE_IDLE
        return True

    return completed_any


def run_simulation(config: OfflineStateConfig, workload_type: str,
                   num_clients: int,
                   pregenerated_workload: tuple = None) -> tuple:
    """
    Run OfflineState simulation.

    Returns (mops, stats_dict) where stats_dict contains:
        local_hits, peer_hits, server_hits (totals across all clients)

    pregenerated_workload: (key_samples, is_read) tuple — required for fair
    comparison; both engines must use the same workload.
    """
    num_ops = max(num_clients * OPS_PER_CLIENT, MIN_OPS)
    key_samples = pregenerated_workload[0][:num_ops]
    is_read = pregenerated_workload[1][:num_ops]

    bloom_size, bloom_hashes = config.effective_bloom_params(num_clients)
    num_groups = min(config.num_peer_groups, num_clients)
    clients = [
        OfflineClient(i, config.max_cache_entries,
                      bloom_size, bloom_hashes,
                      group_id=i % num_groups)
        for i in range(num_clients)
    ]

    # Create GIBF
    gibf_size, gibf_hashes = config.gibf_params(num_clients)
    gibf = GroupIndexedBloomFilter(gibf_size, gibf_hashes, num_groups)

    # Pre-group clients for O(group_size) peer lookups instead of O(N)
    clients_by_group = defaultdict(list)
    for c in clients:
        clients_by_group[c.group_id].append(c)

    ops_per_client = num_ops // num_clients
    op_indices = [[i * ops_per_client] for i in range(num_clients)]
    wl_keys = []
    wl_reads = []
    for i in range(num_clients):
        start = i * ops_per_client
        end = start + ops_per_client
        wl_keys.append(key_samples[start:end])
        wl_reads.append(is_read[start:end])

    total_ticks = int(config.sim_duration_us / config.rdma_rtt_us)
    client_order = np.arange(num_clients)

    start_time = time.monotonic()

    for tick in range(total_ticks):
        # Offline sync at configured intervals
        if tick > 0 and tick % config.sync_interval_ticks == 0:
            run_offline_sync(clients, gibf=gibf)

        np.random.shuffle(client_order)

        for c_idx in client_order:
            client = clients[c_idx]
            tick_client(
                client, clients, config,
                wl_keys[c_idx], wl_reads[c_idx], op_indices[c_idx],
                current_tick=tick,
                clients_by_group=clients_by_group,
            )

    wall_time = time.monotonic() - start_time
    total_ops = sum(c.ops_completed for c in clients)
    total_local = sum(c.local_hits for c in clients)
    total_peer = sum(c.peer_hits for c in clients)
    total_server = sum(c.server_hits for c in clients)
    total_rdma = sum(c.rdma_calls for c in clients)
    total_fp = sum(c.false_positives for c in clients)
    total_group_queries = sum(c.group_peer_queries for c in clients)

    simulated_time_seconds = total_ticks * config.rdma_rtt_us / 1_000_000
    mops = total_ops / simulated_time_seconds / 1_000_000
    rdma_per_op = total_rdma / total_ops if total_ops > 0 else 0

    # Peers queried per peer-read: GIBF queries target group_size peers,
    # broadcast queries target N-1 peers.
    group_size = num_clients // num_groups if num_groups > 0 else num_clients
    broadcast_peer_reads = total_peer + total_fp - total_group_queries
    peers_queried = (total_group_queries * group_size +
                     broadcast_peer_reads * (num_clients - 1))
    peers_per_peer_read = (peers_queried / (total_peer + total_fp)
                           if (total_peer + total_fp) > 0 else 0)

    print(f"  {workload_type.upper()} | {num_clients} clients | "
          f"{total_ops:,} ops in {total_ticks:,} ticks "
          f"({simulated_time_seconds*1000:.1f}ms sim) | "
          f"{mops:.3f} MOPS | "
          f"local={total_local} peer={total_peer} server={total_server} FP={total_fp} | "
          f"RDMA/op={rdma_per_op:.3f} peers/query={peers_per_peer_read:.1f} | "
          f"wall={wall_time:.1f}s")

    # Collect latency data
    all_read_latencies = []
    all_write_latencies = []
    for c in clients:
        all_read_latencies.extend(c.read_latencies)
        all_write_latencies.extend(c.write_latencies)

    stats = {
        'local_hits': total_local,
        'peer_hits': total_peer,
        'server_hits': total_server,
        'total_ops': total_ops,
        'total_rdma': total_rdma,
        'false_positives': total_fp,
        'read_latencies': all_read_latencies,
        'write_latencies': all_write_latencies,
        'group_peer_queries': total_group_queries,
        'peers_per_peer_read': peers_per_peer_read,
    }

    return mops, stats
