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
from typing import Optional

import numpy as np

from offline_state.config import OfflineStateConfig
from offline_state.bloom_filter import BloomFilter
from offline_state.client import (
    OfflineClient,
    PHASE_IDLE, PHASE_PEER_READ,
    PHASE_SERVER_READ, PHASE_SERVER_WRITE,
)

# Reuse workload generation from rcuckoo
from rcuckoo.workload import generate_workload


def _any_peer_has_key(clients, requesting_client_id, key):
    """Check if any peer client has the key in its cache (simulated broadcast)."""
    for c in clients:
        if c.client_id == requesting_client_id:
            continue
        if key in c.cache:
            return True
    return False


def run_offline_sync(clients):
    """
    Offline sync worker: rebuild bloom filters and merge.

    1. Each client rebuilds its local bloom filter from current cache keys
    2. Compute bitwise OR of all local bloom filters
    3. Write result as each client's peer bloom filter
    """
    for c in clients:
        c.local_bloom.rebuild_from_keys(c.cache.keys())

    local_blooms = [c.local_bloom for c in clients]
    merged = BloomFilter.bitwise_or(local_blooms)

    for c in clients:
        c.peer_bloom = merged.copy()


def tick_client(client, clients, config, workload_keys, workload_is_read,
                op_idx_holder):
    """
    Advance one client by one tick.

    Local cache hits are FREE (0 RTT) — the client loops and processes
    multiple local hits within a single tick, since local memory access
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

        if is_read:
            client.op_type = 'read'
            # Check local cache first (0 RTT, 0 RDMA calls)
            if client.cache_get(key) is not None:
                client.ops_completed += 1
                client.local_hits += 1
                completed_any = True
                continue  # FREE — try next op in same tick

            # Check peer bloom filter
            if client.peer_bloom.contains(key):
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
        # Broadcast to peers (1 tick, same rack)
        client.rdma_calls += 1
        if _any_peer_has_key(clients, client.client_id, client.op_key):
            client.cache_put(client.op_key, client.op_key)
            client.ops_completed += 1
            client.peer_hits += 1
            client.phase = PHASE_IDLE
            return True
        else:
            # False positive — fall through to server next tick
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
        client.phase = PHASE_IDLE
        return True

    return completed_any


def run_simulation(config: OfflineStateConfig, workload_type: str,
                   num_clients: int,
                   bloom_fill_callback=None,
                   pregenerated_workload=None) -> tuple:
    """
    Run OfflineState simulation.

    Returns (mops, stats_dict) where stats_dict contains:
        local_hits, peer_hits, server_hits (totals across all clients)
    """
    num_ops = max(num_clients * 50000, 200000)
    if pregenerated_workload is not None:
        key_samples = pregenerated_workload[0][:num_ops]
        is_read = pregenerated_workload[1][:num_ops]
    else:
        key_samples, is_read = generate_workload(
            workload_type, config.total_entries, config.zipf_theta, num_ops
        )

    clients = [
        OfflineClient(i, config.max_cache_entries,
                      config.bloom_filter_size_bits,
                      config.bloom_filter_num_hashes)
        for i in range(num_clients)
    ]

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
            run_offline_sync(clients)
            if bloom_fill_callback is not None:
                avg_local = np.mean([c.local_bloom.fill_rate() for c in clients])
                avg_peer = np.mean([c.peer_bloom.fill_rate() for c in clients])
                avg_cache_size = np.mean([len(c.cache) for c in clients])
                bloom_fill_callback(tick, avg_local, avg_peer, avg_cache_size)

        np.random.shuffle(client_order)

        # No NIC bandwidth constraint — only RTT latency matters.
        # Each client has its own NIC; there is no shared bottleneck.
        for c_idx in client_order:
            client = clients[c_idx]
            tick_client(
                client, clients, config,
                wl_keys[c_idx], wl_reads[c_idx], op_indices[c_idx]
            )

    wall_time = time.monotonic() - start_time
    total_ops = sum(c.ops_completed for c in clients)
    total_local = sum(c.local_hits for c in clients)
    total_peer = sum(c.peer_hits for c in clients)
    total_server = sum(c.server_hits for c in clients)
    total_rdma = sum(c.rdma_calls for c in clients)

    simulated_time_seconds = total_ticks * config.rdma_rtt_us / 1_000_000
    mops = total_ops / simulated_time_seconds / 1_000_000
    rdma_per_op = total_rdma / total_ops if total_ops > 0 else 0

    print(f"  {workload_type.upper()} | {num_clients} clients | "
          f"{total_ops:,} ops in {total_ticks:,} ticks "
          f"({simulated_time_seconds*1000:.1f}ms sim) | "
          f"{mops:.3f} MOPS | "
          f"local={total_local} peer={total_peer} server={total_server} | "
          f"RDMA/op={rdma_per_op:.3f} | "
          f"wall={wall_time:.1f}s")

    stats = {
        'local_hits': total_local,
        'peer_hits': total_peer,
        'server_hits': total_server,
        'total_ops': total_ops,
        'total_rdma': total_rdma,
    }

    return mops, stats
