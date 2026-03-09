"""
Simulation engine for RCuckoo.

Tick-based event-driven simulation where each tick = 1 RDMA RTT.
All clients advance in parallel per tick, subject to NIC capacity
constraints (bandwidth and CAS rate from Section 2.2, Figures 1a-c).
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
from rcuckoo.workload import generate_workload, prepopulate


# Bandwidth consumed per RDMA phase.
#
# Section 2.2, Figure 1b: reads/writes and CAS use independent NIC resources
# (DMA engine vs atomics engine). CAS operations on device memory do NOT
# consume main bandwidth. Only data reads/writes use link bandwidth.
#
# Read: covering read of ~256 bytes (Section 3.2.1)
# Update lock: CAS(device mem, 0 bw) + read rows(256 bw) batched together
# Update lock retry: CAS only (device mem, 0 bw) - Section 3.4.1: "clients spin"
# Update write: write row(73 bw) + CAS release(device mem, 0 bw)
# Insert phases follow the same pattern as update
PHASE_DATA_BYTES = {
    PHASE_READ_ISSUED: 256,       # covering read
    PHASE_UPDATE_LOCK: 256,       # read rows (batched with CAS on first attempt)
    PHASE_UPDATE_WRITE: 73,       # write row
    PHASE_INSERT_LOCK: 256,       # read rows
    PHASE_INSERT_SEARCH: 0,       # local computation
    PHASE_INSERT_WRITE: 73,       # write row
}

# CAS retries: spinning on lock, no data read
# Section 3.4.1: "Clients continuously spin on lock acquisitions"
PHASE_RETRY_BYTES = 0  # CAS uses device memory, not link bandwidth

PHASE_NEEDS_CAS = {
    PHASE_UPDATE_LOCK, PHASE_UPDATE_WRITE,
    PHASE_INSERT_LOCK, PHASE_INSERT_WRITE,
}


def _start_new_op(client, table, lock_table, config, workload_keys,
                  workload_is_read, op_idx_holder):
    """Pick next operation for a client (client-local, no RDMA cost)."""
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

    Returns True if an operation completed this tick.

    IDLE is not a phase that consumes an RTT - when a client finishes
    an operation, it immediately picks the next one and starts its
    first RDMA phase in the same tick.
    """

    if client.phase == PHASE_IDLE:
        _start_new_op(client, table, lock_table, config,
                      workload_keys, workload_is_read, op_idx_holder)

    if client.phase == PHASE_READ_ISSUED:
        # Section 3.2.1: "issue RDMA reads for both rows simultaneously"
        client.cache_row(client.L1, table.keys[client.L1], table.values[client.L1])
        client.cache_row(client.L2, table.keys[client.L2], table.values[client.L2])
        client.phase = PHASE_IDLE
        client.ops_completed += 1
        return True

    if client.phase == PHASE_UPDATE_LOCK:
        # Section 3.2.2: lock acquisition via MCAS
        group = client.mcas_groups[client.mcas_group_idx]
        if lock_table.try_acquire_mcas(group, client.client_id):
            client.acquired_locks.extend(group)
            client.mcas_group_idx += 1
            if client.mcas_group_idx >= len(client.mcas_groups):
                # All locks acquired - read rows in same batch
                # "the client issues read(s) for the corresponding rows
                # immediately afterwards but in the same batch"
                client.cache_row(client.L1, table.keys[client.L1],
                                 table.values[client.L1])
                client.cache_row(client.L2, table.keys[client.L2],
                                 table.values[client.L2])
                client.phase = PHASE_UPDATE_WRITE
        else:
            # "clients retry acquisitions until they succeed"
            client.lock_retries += 1
            if client.lock_retries > 200:
                # Abandon - do not count as completed
                lock_table.release_all(client.acquired_locks)
                client.acquired_locks = []
                client.phase = PHASE_IDLE
        return False

    elif client.phase == PHASE_UPDATE_WRITE:
        # Section 3.2.2: write updated entry, recompute version, release locks
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
        # Section 3.2.3: lock acquisition for insert
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
            if client.lock_retries > 200:
                lock_table.release_all(client.acquired_locks)
                client.acquired_locks = []
                client.phase = PHASE_IDLE
        return False

    elif client.phase == PHASE_INSERT_SEARCH:
        # Section 3.2.3: "the client confirms that the speculative path
        # remains valid"
        key = client.op_key
        L1, L2 = client.L1, client.L2

        # Key already exists
        if table.find_key_in_row(L1, key) >= 0 or table.find_key_in_row(L2, key) >= 0:
            lock_table.release_all(client.acquired_locks)
            client.acquired_locks = []
            client.phase = PHASE_IDLE
            client.ops_completed += 1
            return True

        # Try direct insert into L1 or L2
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

        # BFS cuckoo path search
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
        # Section 3.2.3: write + unlock
        lock_table.release_all(client.acquired_locks)
        client.acquired_locks = []
        client.phase = PHASE_IDLE
        client.ops_completed += 1
        return True

    return False


def run_simulation(config: RCuckooConfig, workload_type: str,
                   num_clients: int,
                   shared_table: Optional[IndexTable] = None) -> float:
    """
    Run the tick-based simulation and return throughput in MOPS.

    Each tick = 1 RDMA RTT. All clients advance in parallel per tick,
    subject to NIC capacity constraints.

    NIC model (Section 2.2, Figures 1a-c):
    - Bandwidth: 100 Gbps = 12,500 bytes/us
    - CAS rate: ~50 MOPS on device memory (Figure 1b)
    - These create natural throughput saturation at high client counts

    If shared_table is provided, reuses it instead of creating a new one.
    """
    if shared_table is not None:
        table = shared_table
    else:
        table = IndexTable(config.num_rows, config.entries_per_row)
        prepopulate(table, config, config.prepopulate_fill)

    lock_table = LockTable(config.num_locks)

    num_ops = max(num_clients * 50000, 200000)
    key_samples, is_read = generate_workload(
        workload_type, table.total_entries, config.zipf_theta, num_ops
    )

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

    # NIC capacity per tick (Section 2.2, Figures 1a-c)
    # Each tick = 1 RDMA RTT. NIC rates are per-us, scale by RTT duration.
    bw_budget = config.nic_bandwidth_bytes_per_us * config.rdma_rtt_us
    cas_budget = config.nic_max_cas_ops_per_us * config.rdma_rtt_us

    total_ticks = int(config.sim_duration_us / config.rdma_rtt_us)
    client_order = np.arange(num_clients)

    start_time = time.monotonic()

    for _tick in range(total_ticks):
        # Two-pass processing per tick:
        # Pass 1: lock-releasing phases (WRITE) - frees locks for others
        # Pass 2: everything else (reads, lock acquisitions, idle)
        np.random.shuffle(client_order)

        bw_remaining = bw_budget
        cas_remaining = cas_budget

        # Pass 1: release locks first
        processed = set()
        for c_idx in client_order:
            client = clients[c_idx]
            if client.phase not in (PHASE_UPDATE_WRITE, PHASE_INSERT_WRITE):
                continue
            op_bytes = PHASE_DATA_BYTES.get(client.phase, 73)
            if bw_remaining < op_bytes:
                continue
            if cas_remaining <= 0:
                continue
            tick_client(
                client, table, lock_table, config,
                wl_keys[c_idx], wl_reads[c_idx], op_indices[c_idx]
            )
            bw_remaining -= op_bytes
            cas_remaining -= 1
            processed.add(c_idx)

        # Pass 2: all other clients
        for c_idx in client_order:
            if c_idx in processed:
                continue
            client = clients[c_idx]
            phase = client.phase

            if phase in (PHASE_UPDATE_WRITE, PHASE_INSERT_WRITE):
                continue

            if phase == PHASE_IDLE:
                idx = op_indices[c_idx][0]
                if idx >= len(wl_reads[c_idx]):
                    idx = 0
                next_is_read = wl_reads[c_idx][idx]
                if next_is_read:
                    op_bytes = config.read_threshold_bytes
                    needs_cas = False
                else:
                    op_bytes = config.read_threshold_bytes
                    needs_cas = True
            else:
                needs_cas = phase in PHASE_NEEDS_CAS
                if (phase in (PHASE_UPDATE_LOCK, PHASE_INSERT_LOCK)
                        and client.lock_retries > 0):
                    op_bytes = PHASE_RETRY_BYTES
                else:
                    op_bytes = PHASE_DATA_BYTES.get(phase, 128)

            if bw_remaining < op_bytes:
                continue
            if needs_cas and cas_remaining <= 0:
                continue

            tick_client(
                client, table, lock_table, config,
                wl_keys[c_idx], wl_reads[c_idx], op_indices[c_idx]
            )

            bw_remaining -= op_bytes
            if needs_cas:
                cas_remaining -= 1

    wall_time = time.monotonic() - start_time
    total_ops = sum(c.ops_completed for c in clients)

    simulated_time_seconds = total_ticks * config.rdma_rtt_us / 1_000_000
    mops = total_ops / simulated_time_seconds / 1_000_000

    print(f"  {workload_type.upper()} | {num_clients} clients | "
          f"{total_ops:,} ops in {total_ticks:,} ticks "
          f"({simulated_time_seconds*1000:.1f}ms sim) | "
          f"{mops:.3f} MOPS | wall={wall_time:.1f}s")

    return mops
