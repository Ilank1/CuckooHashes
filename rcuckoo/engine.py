"""
Simulation engine for RCuckoo.

Tick-based simulation where each tick = 1 base RDMA RTT (~1us).
All server-bound phases take server_rtt_ticks ticks (cross-rack latency).
Lock contention is the primary throughput limiter for write-heavy workloads.
"""

import time
from typing import Optional

import numpy as np

from rcuckoo.config import RCuckooConfig
from rcuckoo.hashing import EMPTY_KEY, compute_locations
from rcuckoo.table import IndexTable, LockTable
from rcuckoo.client import (
    ClientState,
    PHASE_IDLE, PHASE_READ_ISSUED,
    PHASE_UPDATE_LOCK, PHASE_UPDATE_WRITE,
    PHASE_INSERT_LOCK, PHASE_INSERT_SEARCH, PHASE_INSERT_WRITE,
)
from rcuckoo.cuckoo import (
    group_locks_for_mcas, bfs_search_locked, execute_cuckoo_path,
)
from rcuckoo.workload import prepopulate
from sim_config import OPS_PER_CLIENT, MIN_OPS, LOCK_RETRY_LIMIT


def _start_new_op(client, table, lock_table, config, workload_keys,
                  workload_is_read, op_idx_holder):
    idx = op_idx_holder[0]
    if idx >= len(workload_keys):
        idx = 0
    op_idx_holder[0] = idx + 1

    key = int(workload_keys[idx])
    if key == int(EMPTY_KEY):
        key = 0
    client.op_key = key
    is_read = workload_is_read[idx]

    L1, L2 = compute_locations(
        key, config.num_rows, config.locality_f,
        config.hash_salt_1, config.hash_salt_2, config.hash_salt_3
    )
    client.L1 = L1
    client.L2 = L2

    if is_read:
        client.op_type = 'read'
        client.phase = PHASE_READ_ISSUED
    else:
        client.op_type = 'update'
        lock1 = lock_table.row_to_lock(L1, config.rows_per_lock)
        lock2 = lock_table.row_to_lock(L2, config.rows_per_lock)
        client.lock_indices = sorted(set([lock1, lock2]))
        client.acquired_locks = []
        client.mcas_groups = group_locks_for_mcas(
            client.lock_indices, config.locks_per_mcas
        )
        client.mcas_group_idx = 0
        client.phase = PHASE_UPDATE_LOCK
        client.lock_retries = 0


def tick_client(client, table, lock_table, config,
                workload_keys, workload_is_read, op_idx_holder):
    """
    Advance one client by one RDMA round trip.

    IDLE is not a phase that consumes an RTT — when a client finishes
    an operation, it immediately picks the next one and starts its
    first RDMA phase in the same tick.
    """

    if client.phase == PHASE_IDLE:
        _start_new_op(client, table, lock_table, config,
                      workload_keys, workload_is_read, op_idx_holder)

    if client.phase == PHASE_READ_ISSUED:
        if client.ticks_remaining == 0:
            client.ticks_remaining = config.server_rtt_ticks
            client.rdma_calls += 1
        client.ticks_remaining -= 1
        if client.ticks_remaining > 0:
            return False
        client.cache_row(client.L1, table.keys[client.L1], table.values[client.L1])
        client.cache_row(client.L2, table.keys[client.L2], table.values[client.L2])
        client.phase = PHASE_IDLE
        client.ops_completed += 1
        return True

    if client.phase == PHASE_UPDATE_LOCK:
        if client.ticks_remaining == 0:
            client.ticks_remaining = config.server_rtt_ticks
            client.rdma_calls += 1
        client.ticks_remaining -= 1
        if client.ticks_remaining > 0:
            return False
        group = client.mcas_groups[client.mcas_group_idx]
        if lock_table.try_acquire_mcas(group, client.client_id):
            client.acquired_locks.extend(group)
            client.mcas_group_idx += 1
            if client.mcas_group_idx >= len(client.mcas_groups):
                client.cache_row(client.L1, table.keys[client.L1],
                                 table.values[client.L1])
                client.cache_row(client.L2, table.keys[client.L2],
                                 table.values[client.L2])
                client.phase = PHASE_UPDATE_WRITE
        else:
            client.lock_retries += 1
            if client.lock_retries > LOCK_RETRY_LIMIT:
                lock_table.release_all(client.acquired_locks)
                client.acquired_locks = []
                client.phase = PHASE_IDLE
        return False

    elif client.phase == PHASE_UPDATE_WRITE:
        if client.ticks_remaining == 0:
            client.ticks_remaining = config.server_rtt_ticks
            client.rdma_calls += 1
        client.ticks_remaining -= 1
        if client.ticks_remaining > 0:
            return False
        key = client.op_key
        slot = table.find_key_in_row(client.L1, key)
        if slot >= 0:
            table.values[client.L1, slot] = np.uint32((key + 1) & 0xFFFFFFFF)
            table.increment_version(client.L1)
        else:
            slot = table.find_key_in_row(client.L2, key)
            if slot >= 0:
                table.values[client.L2, slot] = np.uint32((key + 1) & 0xFFFFFFFF)
                table.increment_version(client.L2)
        lock_table.release_all(client.acquired_locks)
        client.acquired_locks = []
        client.phase = PHASE_IDLE
        client.ops_completed += 1
        return True

    elif client.phase == PHASE_INSERT_LOCK:
        if client.ticks_remaining == 0:
            client.ticks_remaining = config.server_rtt_ticks
            client.rdma_calls += 1
        client.ticks_remaining -= 1
        if client.ticks_remaining > 0:
            return False
        group = client.mcas_groups[client.mcas_group_idx]
        if lock_table.try_acquire_mcas(group, client.client_id):
            client.acquired_locks.extend(group)
            client.mcas_group_idx += 1
            if client.mcas_group_idx >= len(client.mcas_groups):
                client.locked_rows = set()
                for lock_idx in client.acquired_locks:
                    real_idx = lock_idx % lock_table.num_locks
                    start = real_idx * config.rows_per_lock
                    for r in range(start, min(start + config.rows_per_lock,
                                              config.num_rows)):
                        client.locked_rows.add(r)
                        client.cache_row(r, table.keys[r], table.values[r])
                client.phase = PHASE_INSERT_SEARCH
        else:
            client.lock_retries += 1
            if client.lock_retries > LOCK_RETRY_LIMIT:
                lock_table.release_all(client.acquired_locks)
                client.acquired_locks = []
                client.phase = PHASE_IDLE
        return False

    elif client.phase == PHASE_INSERT_SEARCH:
        key = client.op_key
        L1, L2 = client.L1, client.L2

        if table.find_key_in_row(L1, key) >= 0 or table.find_key_in_row(L2, key) >= 0:
            lock_table.release_all(client.acquired_locks)
            client.acquired_locks = []
            client.phase = PHASE_IDLE
            client.ops_completed += 1
            return True

        slot = table.empty_slot_in_row(L1)
        if slot >= 0:
            table.keys[L1, slot] = np.uint32(key)
            table.values[L1, slot] = np.uint32(client.op_value)
            table.increment_version(L1)
            table.total_entries += 1
            client.phase = PHASE_INSERT_WRITE
            return False

        slot = table.empty_slot_in_row(L2)
        if slot >= 0:
            table.keys[L2, slot] = np.uint32(key)
            table.values[L2, slot] = np.uint32(client.op_value)
            table.increment_version(L2)
            table.total_entries += 1
            client.phase = PHASE_INSERT_WRITE
            return False

        path = bfs_search_locked(table, config, key, L1, L2, client.locked_rows)
        if path:
            execute_cuckoo_path(table, path)
            first_move = path[0]
            table.keys[first_move['from_row'], first_move['from_slot']] = np.uint32(key)
            table.values[first_move['from_row'], first_move['from_slot']] = np.uint32(
                client.op_value)
            table.increment_version(first_move['from_row'])
            table.total_entries += 1
            client.phase = PHASE_INSERT_WRITE
        else:
            lock_table.release_all(client.acquired_locks)
            client.acquired_locks = []
            client.phase = PHASE_IDLE
            client.ops_completed += 1
        return False

    elif client.phase == PHASE_INSERT_WRITE:
        if client.ticks_remaining == 0:
            client.ticks_remaining = config.server_rtt_ticks
            client.rdma_calls += 1
        client.ticks_remaining -= 1
        if client.ticks_remaining > 0:
            return False
        lock_table.release_all(client.acquired_locks)
        client.acquired_locks = []
        client.phase = PHASE_IDLE
        client.ops_completed += 1
        return True

    return False


def run_simulation(config: RCuckooConfig, workload_type: str,
                   num_clients: int,
                   shared_table: Optional[IndexTable] = None,
                   pregenerated_workload=None) -> tuple:
    """Run the tick-based simulation and return (mops, stats_dict)."""
    if shared_table is not None:
        table = shared_table
    else:
        table = IndexTable(config.num_rows, config.entries_per_row)
        prepopulate(table, config, config.prepopulate_fill)

    lock_table = LockTable(config.num_locks)

    num_ops = max(num_clients * OPS_PER_CLIENT, MIN_OPS)
    key_samples = pregenerated_workload[0][:num_ops]
    is_read = pregenerated_workload[1][:num_ops]

    clients = [ClientState(i, config.cache_rows) for i in range(num_clients)]

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

    for _tick in range(total_ticks):
        np.random.shuffle(client_order)

        # Pass 1: release locks first (frees them for other clients)
        processed = set()
        for c_idx in client_order:
            client = clients[c_idx]
            if client.phase not in (PHASE_UPDATE_WRITE, PHASE_INSERT_WRITE):
                continue
            tick_client(
                client, table, lock_table, config,
                wl_keys[c_idx], wl_reads[c_idx], op_indices[c_idx]
            )
            processed.add(c_idx)

        # Pass 2: all other clients
        for c_idx in client_order:
            if c_idx in processed:
                continue
            client = clients[c_idx]
            if client.phase in (PHASE_UPDATE_WRITE, PHASE_INSERT_WRITE):
                continue
            tick_client(
                client, table, lock_table, config,
                wl_keys[c_idx], wl_reads[c_idx], op_indices[c_idx]
            )

    wall_time = time.monotonic() - start_time
    total_ops = sum(c.ops_completed for c in clients)
    total_rdma = sum(c.rdma_calls for c in clients)

    simulated_time_seconds = total_ticks * config.rdma_rtt_us / 1_000_000
    mops = total_ops / simulated_time_seconds / 1_000_000
    rdma_per_op = total_rdma / total_ops if total_ops > 0 else 0

    print(f"  {workload_type.upper()} | {num_clients} clients | "
          f"{total_ops:,} ops in {total_ticks:,} ticks "
          f"({simulated_time_seconds*1000:.1f}ms sim) | "
          f"{mops:.3f} MOPS | RDMA/op={rdma_per_op:.3f} | "
          f"wall={wall_time:.1f}s")

    stats = {
        'total_ops': total_ops,
        'total_rdma': total_rdma,
    }

    return mops, stats
