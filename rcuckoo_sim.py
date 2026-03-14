"""
RCuckoo Simulator - Faithful implementation of the ATC'25 paper:
"Cuckoo for Clients: Disaggregated Cuckoo Hashing" by Stewart Grant and Alex C. Snoeren

Event-driven simulation where each tick = 1 RDMA round trip (~1 us).
All clients advance in parallel per tick, naturally modeling RDMA parallelism.

Key design: In real RDMA, each client independently sends requests to the NIC
which processes them in parallel. More clients = more parallel operations =
higher aggregate throughput. We model this by advancing all clients each tick.
"""

import time
import numpy as np
from collections import deque
from dataclasses import dataclass, field
from typing import Optional

# =============================================================================
# CONFIGURATION - All parameters from the paper with citations
# =============================================================================

@dataclass
class RCuckooConfig:
    # --- Table parameters (Section 3.1, Section 5.3) ---
    # Paper: "100-M-entry table" with "8 entries per row"
    total_entries: int = 100_000_000
    entries_per_row: int = 8  # Paper Section 3.1: "we use 8 in practice"

    # Paper Section 5.3: "32-bit key and 32-bit value"
    key_size_bits: int = 32
    value_size_bits: int = 32

    # --- Hashing (Section 3.3) ---
    # Paper: "we set f = 2.3 based on this empirical data"
    locality_f: float = 2.3
    # Paper: "We use xxHash in our implementation" - we use integer mixing
    hash_salt_1: int = 0xDEADBEEF
    hash_salt_2: int = 0xCAFEBABE
    hash_salt_3: int = 0x12345678

    # --- Locking (Section 3.4) ---
    # Paper: "16 rows per lock" (Section 3.4.1)
    rows_per_lock: int = 16
    # Paper: "single-bit locks correspond to 64 locks per message"
    locks_per_mcas: int = 64

    # --- Search (Section 3.2.3) ---
    # Paper: "maximum search depth (we use a depth of five)"
    max_search_depth: int = 5

    # --- Client cache (Section 3.2.3, Section 5.1) ---
    # Paper: "64-KB client index-table caches"
    client_cache_size_bytes: int = 64 * 1024

    # --- Pre-population (Section 5.3) ---
    # Paper: "pre-populate it with 90 M entries" => 90% fill
    prepopulate_fill: float = 0.9

    # --- Workload (Section 5.3) ---
    # Paper: "Zipf(0.99) distribution"
    zipf_theta: float = 0.99

    # --- RDMA timing (Section 1, Section 2.2) ---
    # Paper: "RDMA latency is approximately 1 us" (NIC processing time).
    # ESTIMATION: Effective per-operation RTT including network traversal,
    # doorbell posting, and CQ polling is ~3us in practice (derived from
    # paper Figure 6 YCSB-C: 10 clients → 2.99 MOPS → ~3.3us/op/client).
    rdma_rtt_us: float = 3.0

    # --- NIC capacity (Section 2.2, Figures 1a-c) ---
    # Paper: "100-Gbps ConnectX-5 testbed hardware" (Section 2.2)
    nic_bandwidth_gbps: float = 100.0
    # Paper Figure 1b: reads/writes scale to ~75 MOPS, CAS ~50 MOPS (device memory)
    # Paper Figure 1c: contended CAS on device memory ~16 MOPS (single address ~3 MOPS)
    nic_max_read_ops_per_us: int = 75
    nic_max_cas_ops_per_us: int = 50  # device memory, independent addresses
    # Paper Section 3.2.1: covering read threshold
    # Researcher notes: read_threshold_bytes = 256
    read_threshold_bytes: int = 256

    # --- Figure 6 client counts (from paper Figure 6 x-axis) ---
    # Validated against researcher data: [8, 16, 40, 80, 160, 320]
    figure6_client_counts: list = field(
        default_factory=lambda: [8, 16, 40, 80, 160, 320]
    )

    # --- Simulation ---
    # ESTIMATION: simulation duration in simulated microseconds
    sim_duration_us: int = 100_000  # 100ms of simulated time
    num_trials: int = 1

    @property
    def num_rows(self) -> int:
        return self.total_entries // self.entries_per_row

    @property
    def entry_size_bytes(self) -> int:
        return (self.key_size_bits + self.value_size_bits) // 8

    @property
    def row_size_bytes(self) -> int:
        # entries + 1-byte version + 8-byte CRC (Section 3.1)
        return self.entries_per_row * self.entry_size_bytes + 1 + 8

    @property
    def num_locks(self) -> int:
        return (self.num_rows + self.rows_per_lock - 1) // self.rows_per_lock

    @property
    def cache_rows(self) -> int:
        return self.client_cache_size_bytes // self.row_size_bytes

    @property
    def nic_bandwidth_bytes_per_us(self) -> float:
        """NIC bandwidth in bytes per microsecond (= per tick)."""
        return self.nic_bandwidth_gbps * 1e9 / 8 / 1e6  # 100 Gbps = 12500 B/us


# =============================================================================
# HASH FUNCTIONS (Section 3.3)
# =============================================================================

EMPTY_KEY = np.uint32(0xFFFFFFFF)


def _hash_with_salt(key: int, salt: int) -> int:
    """Fast integer hash mixing (substitute for xxHash)."""
    h = (key ^ salt) & 0xFFFFFFFF
    h = (((h >> 16) ^ h) * 0x45d9f3b) & 0xFFFFFFFF
    h = (((h >> 16) ^ h) * 0x45d9f3b) & 0xFFFFFFFF
    h = ((h >> 16) ^ h) & 0xFFFFFFFF
    return h


def compute_locations(key: int, num_rows: int, f: float,
                      salt1: int, salt2: int, salt3: int) -> tuple:
    """
    Compute two cuckoo hash locations for a key (Section 3.3).

    Paper formula:
        L1(K) = h1(K) mod T
        L2(K) = (L1 + (h2(K) mod floor(f^(f + Z(h3(K)))))) mod T

    T = num_rows, Z(x) = number of trailing zeros in x, f = locality parameter.
    """
    h1 = _hash_with_salt(key, salt1)
    h2 = _hash_with_salt(key, salt2)
    h3 = _hash_with_salt(key, salt3)

    L1 = h1 % num_rows

    # Z(x) = number of trailing zeros
    if h3 == 0:
        z = 32
    else:
        z = (h3 & (-h3)).bit_length() - 1

    # offset = h2(K) mod floor(f^(f + Z(h3(K))))
    exponent = f + z
    max_offset = int(f ** exponent)
    max_offset = min(max_offset, num_rows)
    if max_offset < 1:
        max_offset = 1

    offset = h2 % max_offset
    if offset == 0:
        offset = 1  # Ensure L1 != L2

    L2 = (L1 + offset) % num_rows
    return L1, L2


# Vectorized location computation for batch pre-population
def compute_locations_batch(keys: np.ndarray, num_rows: int, f: float,
                            salt1: int, salt2: int, salt3: int):
    """Batch compute locations for numpy array of keys. Fully vectorized."""
    keys_int = keys.astype(np.int64)

    def _batch_hash(k, salt):
        h = (k ^ salt) & 0xFFFFFFFF
        h = (((h >> 16) ^ h) * 0x45d9f3b) & 0xFFFFFFFF
        h = (((h >> 16) ^ h) * 0x45d9f3b) & 0xFFFFFFFF
        h = ((h >> 16) ^ h) & 0xFFFFFFFF
        return h

    h1 = _batch_hash(keys_int, salt1)
    h2 = _batch_hash(keys_int, salt2)
    h3 = _batch_hash(keys_int, salt3)

    L1 = h1 % num_rows

    # Vectorized trailing zeros: Z(x) = ctz(x)
    # lsb = x & (-x) isolates lowest set bit; log2(lsb) = ctz
    h3_signed = h3.astype(np.int64)
    lsb = h3_signed & (-h3_signed)
    z_arr = np.zeros(len(keys), dtype=np.int64)
    nonzero = h3 != 0
    z_arr[nonzero] = np.log2(lsb[nonzero].astype(np.float64)).astype(np.int64)
    z_arr[~nonzero] = 32

    exponents = f + z_arr
    max_offsets = np.minimum(np.floor(f ** exponents).astype(np.int64), num_rows)
    max_offsets = np.maximum(max_offsets, 1)

    offsets = np.array(h2) % max_offsets
    offsets[offsets == 0] = 1

    L2 = (np.array(L1) + offsets) % num_rows
    return np.array(L1, dtype=np.int64), np.array(L2, dtype=np.int64)


# =============================================================================
# DATA STRUCTURES (Section 3.1) - NumPy-backed for performance
# =============================================================================

class IndexTable:
    """
    Remote index table (Section 3.1).

    Paper: "a single region of RDMA-registered main memory divided into rows
    of fixed-width entries. Each row contains n associative entries and
    terminates with an 8-bit version number and 64-bit CRC."

    Backed by numpy arrays for fast pre-population and access.
    """

    def __init__(self, num_rows: int, entries_per_row: int):
        self.num_rows = num_rows
        self.entries_per_row = entries_per_row
        # Keys and values as 2D arrays: [num_rows, entries_per_row]
        self.keys = np.full((num_rows, entries_per_row), EMPTY_KEY, dtype=np.uint32)
        self.values = np.zeros((num_rows, entries_per_row), dtype=np.uint32)
        # Version numbers (8-bit per row)
        self.versions = np.zeros(num_rows, dtype=np.uint8)
        self.total_entries = 0

    def find_key_in_row(self, row: int, key: int) -> int:
        """Return slot index of key in row, or -1."""
        for s in range(self.entries_per_row):
            if self.keys[row, s] == key:
                return s
        return -1

    def empty_slot_in_row(self, row: int) -> int:
        """Return first empty slot index in row, or -1."""
        for s in range(self.entries_per_row):
            if self.keys[row, s] == EMPTY_KEY:
                return s
        return -1

    def increment_version(self, row: int):
        """Increment row version (Section 3.1: 8-bit version number)."""
        self.versions[row] = (self.versions[row] + 1) & 0xFF


class LockTable:
    """
    Lock table in NIC device memory (Section 3.4).

    Paper: "Locks (stored in a bit vector in NIC memory) each protect a tunable
    number (here, two; 16 in our experiments) of index rows."

    In simulated-time model, locks are a simple boolean array. Contention is
    modeled by checking lock state each tick.
    """

    def __init__(self, num_locks: int):
        self.num_locks = num_locks
        self.locked = np.zeros(num_locks, dtype=np.bool_)
        # Track which client holds each lock (for debugging)
        self.holder = np.full(num_locks, -1, dtype=np.int32)

    def row_to_lock(self, row_idx: int, rows_per_lock: int) -> int:
        return row_idx // rows_per_lock

    def try_acquire_mcas(self, lock_indices: list, client_id: int) -> bool:
        """
        Simulate MCAS lock acquisition (Section 3.4.1).

        Paper: "leverages masked CAS (MCAS) operations to obtain up to 64
        locks simultaneously while avoiding false sharing."
        Paper: "To avoid deadlock RCuckoo acquires locks in increasing order."
        """
        indices = sorted(set(idx % self.num_locks for idx in lock_indices))
        # Check all locks are free
        for idx in indices:
            if self.locked[idx]:
                return False
        # All free - acquire atomically
        for idx in indices:
            self.locked[idx] = True
            self.holder[idx] = client_id
        return True

    def release_all(self, lock_indices: list):
        for idx in set(lock_indices):
            real_idx = idx % self.num_locks
            self.locked[real_idx] = False
            self.holder[real_idx] = -1


# =============================================================================
# CLIENT STATE MACHINE (Section 3.2)
# =============================================================================

# Operation phases - each phase consumes 1 RDMA round trip
PHASE_IDLE = 0
PHASE_READ_ISSUED = 1          # Read: RDMA read both rows (1 RTT)
PHASE_UPDATE_LOCK = 2          # Update: try acquire locks (1 RTT)
PHASE_UPDATE_LOCKED_READ = 3   # Update: read rows with locks held (batched with lock)
PHASE_UPDATE_WRITE = 4         # Update: write + release (1 RTT)
PHASE_INSERT_LOCK = 5          # Insert: try acquire locks (1 RTT)
PHASE_INSERT_SEARCH = 6       # Insert: second search in locked rows
PHASE_INSERT_WRITE = 7        # Insert: execute cuckoo path + release


class ClientState:
    """State for one simulated client."""

    __slots__ = [
        'client_id', 'phase', 'op_type', 'op_key', 'op_value',
        'L1', 'L2', 'lock_indices', 'acquired_locks',
        'mcas_groups', 'mcas_group_idx',
        'ops_completed', 'lock_retries',
        'cache', 'cache_order', 'max_cache_rows',
        'cuckoo_path', 'locked_rows'
    ]

    def __init__(self, client_id: int, max_cache_rows: int):
        self.client_id = client_id
        self.phase = PHASE_IDLE
        self.op_type = None  # 'read' or 'update'
        self.op_key = 0
        self.op_value = 0
        self.L1 = 0
        self.L2 = 0
        self.lock_indices = []
        self.acquired_locks = []
        self.mcas_groups = []
        self.mcas_group_idx = 0
        self.ops_completed = 0
        self.lock_retries = 0
        self.cache = {}
        self.cache_order = deque()
        self.max_cache_rows = max_cache_rows
        self.cuckoo_path = None
        self.locked_rows = set()

    def cache_row(self, row_idx: int, keys, values):
        """Cache a row's key/value data."""
        if row_idx in self.cache:
            self.cache[row_idx] = (keys.copy(), values.copy())
            return
        if len(self.cache) >= self.max_cache_rows:
            evict = self.cache_order.popleft()
            self.cache.pop(evict, None)
        self.cache[row_idx] = (keys.copy(), values.copy())
        self.cache_order.append(row_idx)


# =============================================================================
# SIMULATION ENGINE
# =============================================================================

def group_locks_for_mcas(lock_indices: list, locks_per_mcas: int) -> list:
    """
    Group locks into MCAS groups (Section 3.4.1).
    Paper: "clients group the necessary locks into the smallest number of sets
    as possible (where each set is an attempt to acquire one or more locks
    within a single 64-bit span)"
    """
    sorted_locks = sorted(set(lock_indices))
    if not sorted_locks:
        return []
    groups = []
    current_group = [sorted_locks[0]]
    for lock in sorted_locks[1:]:
        if lock - current_group[0] < locks_per_mcas:
            current_group.append(lock)
        else:
            groups.append(current_group)
            current_group = [lock]
    groups.append(current_group)
    return groups


def bfs_search_locked(table: IndexTable, config: RCuckooConfig,
                      key: int, L1: int, L2: int,
                      locked_rows: set) -> Optional[list]:
    """
    BFS search for cuckoo path within locked rows (Section 3.2.3).

    Paper: "We use breadth-first search (BFS) to identify short paths in an
    attempt to minimize bandwidth and locking overhead."
    """
    queue = deque()
    visited = set()
    parent = {}

    # Seed BFS from entries in L1 and L2
    for start_row in [L1, L2]:
        for slot in range(config.entries_per_row):
            if table.keys[start_row, slot] != EMPTY_KEY:
                queue.append((start_row, slot, 0))
                visited.add((start_row, slot))

    while queue:
        row_idx, slot_idx, depth = queue.popleft()
        if depth >= config.max_search_depth:
            continue

        entry_key = int(table.keys[row_idx, slot_idx])
        if entry_key == EMPTY_KEY:
            continue

        # Find alternate location
        eL1, eL2 = compute_locations(
            entry_key, config.num_rows, config.locality_f,
            config.hash_salt_1, config.hash_salt_2, config.hash_salt_3
        )
        alt_row = eL2 if row_idx == eL1 else eL1

        if alt_row not in locked_rows:
            continue

        empty_slot = table.empty_slot_in_row(alt_row)
        if empty_slot >= 0:
            # Found path - reconstruct
            path = [{
                'from_row': row_idx, 'from_slot': slot_idx,
                'to_row': alt_row, 'to_slot': empty_slot,
                'key': entry_key, 'value': int(table.values[row_idx, slot_idx])
            }]
            cur = (row_idx, slot_idx)
            while cur in parent:
                prev_row, prev_slot, cur_row, cur_slot = parent[cur]
                path.append({
                    'from_row': prev_row, 'from_slot': prev_slot,
                    'to_row': cur_row, 'to_slot': cur_slot,
                    'key': int(table.keys[prev_row, prev_slot]),
                    'value': int(table.values[prev_row, prev_slot])
                })
                cur = (prev_row, prev_slot)
            path.reverse()
            return path

        # Continue BFS through alt_row entries
        for s in range(config.entries_per_row):
            if (alt_row, s) not in visited and table.keys[alt_row, s] != EMPTY_KEY:
                visited.add((alt_row, s))
                parent[(alt_row, s)] = (row_idx, slot_idx, alt_row, s)
                queue.append((alt_row, s, depth + 1))

    return None


def execute_cuckoo_path(table: IndexTable, path: list):
    """
    Execute cuckoo path (Section 3.2.3).
    Paper: "cuckooing elements one at a time, starting by moving the last
    entry in the path to the empty location"
    """
    for move in reversed(path):
        fr, fs = move['from_row'], move['from_slot']
        tr, ts = move['to_row'], move['to_slot']
        table.keys[tr, ts] = move['key']
        table.values[tr, ts] = move['value']
        table.increment_version(tr)
        table.keys[fr, fs] = EMPTY_KEY
        table.values[fr, fs] = 0
        table.increment_version(fr)


def _start_new_op(client: ClientState, table: IndexTable, lock_table: LockTable,
                  config: RCuckooConfig, workload_keys: np.ndarray,
                  workload_is_read: np.ndarray, op_idx_holder: list):
    """Pick next operation for a client. This is client-local (no RDMA RTT)."""
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


def tick_client(client: ClientState, table: IndexTable, lock_table: LockTable,
                config: RCuckooConfig, workload_keys: np.ndarray,
                workload_is_read: np.ndarray, op_idx_holder: list) -> bool:
    """
    Advance one client by one RDMA round trip (tick).

    Returns True if an operation was completed this tick.

    Each phase represents one round trip of RDMA communication.
    This models the paper's protocol exactly:
    - Read: 1 RTT (Section 3.2.1)
    - Update: 2 RTTs minimum (Section 3.2.2)
    - Insert: 2+ RTTs (Section 3.2.3)

    IDLE is not a phase that consumes an RTT - it is client-local setup.
    When a client completes an op, it immediately picks the next op and
    starts its first RDMA phase in the same tick.
    """

    if client.phase == PHASE_IDLE:
        # Pick next op (client-local, no RTT cost)
        _start_new_op(client, table, lock_table, config,
                      workload_keys, workload_is_read, op_idx_holder)
        # Fall through to execute first RDMA phase in same tick

    if client.phase == PHASE_READ_ISSUED:
        # --- READ (Section 3.2.1) ---
        # Paper: "issue RDMA reads for both rows simultaneously"
        # Paper: "single round-trip reads for small values"
        # 1 RTT: read both rows
        client.cache_row(client.L1, table.keys[client.L1], table.values[client.L1])
        client.cache_row(client.L2, table.keys[client.L2], table.values[client.L2])
        client.phase = PHASE_IDLE
        client.ops_completed += 1
        return True

    if client.phase == PHASE_UPDATE_LOCK:
        # --- UPDATE lock acquisition (Section 3.2.2) ---
        # Paper: "Due to RCuckoo's dependent hashing, it is usually possible
        # to attempt to acquire both locks in a single MCAS operation"
        # 1 RTT: try MCAS
        group = client.mcas_groups[client.mcas_group_idx]
        if lock_table.try_acquire_mcas(group, client.client_id):
            client.acquired_locks.extend(group)
            client.mcas_group_idx += 1
            if client.mcas_group_idx >= len(client.mcas_groups):
                # All locks acquired - read rows in same batch
                # Paper: "the client issues read(s) for the corresponding rows
                # immediately afterwards but in the same batch"
                client.cache_row(client.L1, table.keys[client.L1],
                                 table.values[client.L1])
                client.cache_row(client.L2, table.keys[client.L2],
                                 table.values[client.L2])
                client.phase = PHASE_UPDATE_WRITE
            # else: need another RTT for next MCAS group
        else:
            # Lock contended - retry next tick
            # Paper: "clients retry acquisitions until they succeed"
            client.lock_retries += 1
            if client.lock_retries > 200:
                # Give up (shouldn't happen normally)
                lock_table.release_all(client.acquired_locks)
                client.acquired_locks = []
                client.phase = PHASE_IDLE
                client.ops_completed += 1
        return False

    elif client.phase == PHASE_UPDATE_WRITE:
        # --- UPDATE write phase (Section 3.2.2) ---
        # 1 RTT: write updated entry + release locks
        # Paper: "the client first writes the updated/freed entry and
        # recomputed row version and CRC before releasing the locks"
        key = client.op_key
        # Find and update
        slot = table.find_key_in_row(client.L1, key)
        if slot >= 0:
            table.values[client.L1, slot] = np.uint32((key + 1) & 0xFFFFFFFF)
            table.increment_version(client.L1)
        else:
            slot = table.find_key_in_row(client.L2, key)
            if slot >= 0:
                table.values[client.L2, slot] = np.uint32((key + 1) & 0xFFFFFFFF)
                table.increment_version(client.L2)
        # Release locks
        lock_table.release_all(client.acquired_locks)
        client.acquired_locks = []
        client.phase = PHASE_IDLE
        client.ops_completed += 1
        return True

    elif client.phase == PHASE_INSERT_LOCK:
        # --- INSERT lock acquisition (Section 3.2.3) ---
        group = client.mcas_groups[client.mcas_group_idx]
        if lock_table.try_acquire_mcas(group, client.client_id):
            client.acquired_locks.extend(group)
            client.mcas_group_idx += 1
            if client.mcas_group_idx >= len(client.mcas_groups):
                # All locks acquired - sync cache
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
                client.ops_completed += 1
        return False

    elif client.phase == PHASE_INSERT_SEARCH:
        # --- INSERT second search (Section 3.2.3) ---
        # Paper: "the client confirms that the speculative path remains valid"
        key = client.op_key
        L1, L2 = client.L1, client.L2

        # Check if key already exists
        if table.find_key_in_row(L1, key) >= 0 or table.find_key_in_row(L2, key) >= 0:
            lock_table.release_all(client.acquired_locks)
            client.acquired_locks = []
            client.phase = PHASE_IDLE
            client.ops_completed += 1
            return True

        # Try direct insert
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

        # Need cuckoo path
        path = bfs_search_locked(table, config, key, L1, L2, client.locked_rows)
        if path:
            execute_cuckoo_path(table, path)
            # Insert new key into freed slot
            first_move = path[0]
            table.keys[first_move['from_row'], first_move['from_slot']] = np.uint32(key)
            table.values[first_move['from_row'], first_move['from_slot']] = np.uint32(
                client.op_value)
            table.increment_version(first_move['from_row'])
            table.total_entries += 1
            client.phase = PHASE_INSERT_WRITE
        else:
            # No path found - release and retry
            lock_table.release_all(client.acquired_locks)
            client.acquired_locks = []
            client.phase = PHASE_IDLE
            client.ops_completed += 1
        return False

    elif client.phase == PHASE_INSERT_WRITE:
        # --- INSERT write + unlock (Section 3.2.3) ---
        lock_table.release_all(client.acquired_locks)
        client.acquired_locks = []
        client.phase = PHASE_IDLE
        client.ops_completed += 1
        return True

    return False


# =============================================================================
# PRE-POPULATION (Section 5.3)
# =============================================================================

def prepopulate(table: IndexTable, config: RCuckooConfig, fill_fraction: float):
    """
    Pre-populate table (Section 5.3).
    Paper: "pre-populate it with 90 M entries that each consist of a 32-bit
    key and 32-bit value"

    Uses batch hash computation and row-fill tracking for speed.
    """
    target = int(config.total_entries * fill_fraction)
    print(f"Pre-populating table: {target:,} entries "
          f"({fill_fraction*100:.0f}% of {config.total_entries:,})...")

    # Track fill level per row (starts at 0, max = entries_per_row)
    row_fill = np.zeros(config.num_rows, dtype=np.int32)
    inserted = 0
    batch_size = 500_000
    key_start = 0
    t0 = time.monotonic()

    while inserted < target:
        batch_end = min(key_start + batch_size, 0xFFFFFFFF)
        keys = np.arange(key_start, batch_end, dtype=np.uint32)
        keys = keys[keys != EMPTY_KEY]

        # Batch compute hash locations
        L1s, L2s = compute_locations_batch(
            keys, config.num_rows, config.locality_f,
            config.hash_salt_1, config.hash_salt_2, config.hash_salt_3
        )

        # Insert keys using row fill tracking
        epr = config.entries_per_row
        for i in range(len(keys)):
            if inserted >= target:
                break
            k = keys[i]
            l1 = int(L1s[i])
            l2 = int(L2s[i])
            if row_fill[l1] < epr:
                slot = row_fill[l1]
                table.keys[l1, slot] = k
                table.values[l1, slot] = k
                row_fill[l1] += 1
                inserted += 1
            elif row_fill[l2] < epr:
                slot = row_fill[l2]
                table.keys[l2, slot] = k
                table.values[l2, slot] = k
                row_fill[l2] += 1
                inserted += 1

        key_start = batch_end
        elapsed = time.monotonic() - t0
        rate = inserted / elapsed if elapsed > 0 else 0
        print(f"\r  {inserted:,}/{target:,} ({inserted/target*100:.0f}%) "
              f"[{rate/1e6:.1f}M keys/s]", end="")

    table.total_entries = inserted
    elapsed = time.monotonic() - t0
    print(f"\r  Pre-population complete: {inserted:,} entries in {elapsed:.1f}s"
          f"                              ")


# =============================================================================
# WORKLOAD GENERATION (Section 5.3)
# =============================================================================

def ycsb_zipf_keys(num_keys: int, theta: float, size: int) -> np.ndarray:
    """
    Generate YCSB-compatible bounded Zipfian key indices over [0, num_keys).

    YCSB Zipf(theta) over N items: P(rank k) ∝ 1/(k+1)^theta for k=0..N-1.
    This is DIFFERENT from np.random.zipf(a) which generates unbounded Zeta.

    Uses the standard YCSB ZipfianGenerator algorithm:
      https://github.com/brianfrankcooper/YCSB (ZipfianGenerator.java)
    """
    N = num_keys
    if abs(theta - 1.0) < 1e-6:
        theta = 0.999

    alpha = 1.0 / (1.0 - theta)

    # Approximate H(N, theta) = sum_{i=1}^{N} 1/i^theta
    # Euler-Maclaurin: H(N,s) ≈ (N^(1-s)-1)/(1-s) + 0.5*(1+N^(-s))
    zeta_n = (N**(1 - theta) - 1) / (1 - theta) + 0.5 * (1 + N**(-theta))
    zeta_2 = 1.0 + 1.0 / 2.0**theta

    eta = (1.0 - (2.0 / N)**(1 - theta)) / (1.0 - zeta_2 / zeta_n)

    u = np.random.random(size)
    uz = u * zeta_n

    result = np.empty(size, dtype=np.int64)
    mask1 = uz < 1.0
    mask2 = (~mask1) & (uz < 1.0 + 0.5**theta)
    mask3 = ~(mask1 | mask2)

    result[mask1] = 0
    result[mask2] = 1
    result[mask3] = np.clip(
        (N * ((eta * u[mask3] - eta + 1.0) ** alpha)).astype(np.int64),
        0, N - 1
    )

    return result


def generate_workload(workload_type: str, num_keys: int, zipf_theta: float,
                      num_ops: int):
    """
    Generate YCSB workload (Section 5.3).

    Paper workloads:
    - YCSB-C: 100% read
    - YCSB-B: 95% read, 5% update
    - YCSB-A: 50% read, 50% update

    Paper: "Zipf(0.99) distribution"
    """
    if workload_type == "ycsb-c":
        read_ratio = 1.0
    elif workload_type == "ycsb-b":
        read_ratio = 0.95
    elif workload_type == "ycsb-a":
        read_ratio = 0.50
    else:
        raise ValueError(f"Unknown workload: {workload_type}")

    # YCSB-style bounded Zipfian over num_keys items
    key_indices = ycsb_zipf_keys(num_keys, zipf_theta, num_ops)
    key_samples = key_indices.astype(np.uint32)

    # Generate read/write decisions
    is_read = np.random.random(num_ops) < read_ratio

    return key_samples, is_read


# =============================================================================
# MAIN SIMULATION LOOP
# =============================================================================

    # Bandwidth consumed per RDMA phase (paper's protocol analysis)
    #
    # Paper Section 2.2, Figure 1b: reads/writes and CAS use independent NIC
    # resources (DMA engine vs atomics engine). CAS operations on device memory
    # do NOT consume main bandwidth. Only data reads/writes use link bandwidth.
    #
    # Read: covering read of ~256 bytes (Section 3.2.1, researcher notes)
    # Update lock (first): CAS(device mem, 0 bw) + read rows(256 bw) batched
    # Update lock (retry): CAS only (device mem, 0 bw) - "clients spin" (Sec 3.4.1)
    # Update write: write row(73 bw) + CAS release(device mem, 0 bw)
    # Insert phases similar to update
PHASE_DATA_BYTES = {
    PHASE_READ_ISSUED: 256,       # covering read
    PHASE_UPDATE_LOCK: 256,       # read rows (batched with CAS on first attempt)
    PHASE_UPDATE_WRITE: 73,       # write row
    PHASE_INSERT_LOCK: 256,       # read rows
    PHASE_INSERT_SEARCH: 0,       # local computation
    PHASE_INSERT_WRITE: 73,       # write row
}

# Bandwidth for CAS retries: just spinning on lock, no data read
# Paper Section 3.4.1: "Clients continuously spin on lock acquisitions"
PHASE_RETRY_BYTES = 0  # CAS uses device memory, not link bandwidth

PHASE_NEEDS_CAS = {
    PHASE_UPDATE_LOCK, PHASE_UPDATE_WRITE,
    PHASE_INSERT_LOCK, PHASE_INSERT_WRITE,
}


def run_simulation(config: RCuckooConfig, workload_type: str,
                   num_clients: int,
                   shared_table: Optional[IndexTable] = None) -> float:
    """
    Run event-driven simulation and return throughput in MOPS.

    Each tick = 1 RDMA RTT. All clients advance in parallel per tick,
    subject to NIC capacity constraints.

    NIC model (Section 2.2, Figures 1a-c):
    - Bandwidth: 100 Gbps = 12,500 bytes/us limits total data per tick
    - CAS rate: ~50 MOPS on device memory (Figure 1b)
    - These create natural throughput saturation at high client counts

    If shared_table is provided, reuses it instead of creating a new one.
    """
    # Create or reuse table
    if shared_table is not None:
        table = shared_table
    else:
        table = IndexTable(config.num_rows, config.entries_per_row)
        prepopulate(table, config, config.prepopulate_fill)

    # Fresh lock table for each run
    lock_table = LockTable(config.num_locks)

    # Generate workload
    num_ops = max(num_clients * 50000, 200000)
    key_samples, is_read = generate_workload(
        workload_type, table.total_entries, config.zipf_theta, num_ops
    )

    # Create clients
    clients = [ClientState(i, config.cache_rows) for i in range(num_clients)]

    # Per-client operation index
    ops_per_client = num_ops // num_clients
    op_indices = [[i * ops_per_client] for i in range(num_clients)]

    # Pre-compute workload slices
    wl_keys = []
    wl_reads = []
    for i in range(num_clients):
        start = i * ops_per_client
        end = start + ops_per_client
        wl_keys.append(key_samples[start:end])
        wl_reads.append(is_read[start:end])

    # NIC capacity per tick (Section 2.2, Figures 1a-c)
    # Each tick = 1 RDMA RTT. NIC rates are per-us, so scale by RTT.
    # E.g., RTT=3us means 3x more NIC capacity per tick (NIC processes
    # other clients' requests during the round trip).
    bw_budget = config.nic_bandwidth_bytes_per_us * config.rdma_rtt_us
    cas_budget = config.nic_max_cas_ops_per_us * config.rdma_rtt_us

    # Simulation loop
    total_ticks = int(config.sim_duration_us / config.rdma_rtt_us)
    client_order = np.arange(num_clients)

    start_time = time.monotonic()

    for _tick in range(total_ticks):
        # Two-pass processing per tick:
        # Pass 1: Clients releasing locks (WRITE phases) - frees locks for others
        # Pass 2: All other clients (reads, lock acquisitions, idle)
        # This models NIC FIFO: releases queued before new requests.
        np.random.shuffle(client_order)

        bw_remaining = bw_budget
        cas_remaining = cas_budget

        # Pass 1: Process lock-releasing phases first (UPDATE_WRITE, INSERT_WRITE)
        # These clients hold locks and need to release them ASAP.
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

        # Pass 2: Process all other clients (skip pass-1 clients)
        for c_idx in client_order:
            if c_idx in processed:
                continue
            client = clients[c_idx]
            phase = client.phase

            # Skip unprocessed WRITE phases (hit budget limit in pass 1)
            if phase in (PHASE_UPDATE_WRITE, PHASE_INSERT_WRITE):
                continue

            # Determine resource needs
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
                # CAS retries (spinning) use device memory, not link bandwidth
                # Paper Section 3.4.1: "Clients continuously spin on lock
                # acquisitions"
                if (phase in (PHASE_UPDATE_LOCK, PHASE_INSERT_LOCK)
                        and client.lock_retries > 0):
                    op_bytes = PHASE_RETRY_BYTES
                else:
                    op_bytes = PHASE_DATA_BYTES.get(phase, 128)

            # Check bandwidth budget
            if bw_remaining < op_bytes:
                continue

            # Check CAS budget
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


# =============================================================================
# FIGURE 6 EVALUATION (Section 5.3)
# =============================================================================

def run_figure6(config: RCuckooConfig):
    """
    Reproduce Figure 6 (Section 5.3).

    Paper: "Figure 6 shows YCSB throughput for RCuckoo, FUSEE, Clover, and
    Sherman on three different YCSB workloads."
    """
    workloads = ["ycsb-c", "ycsb-b", "ycsb-a"]
    client_counts = config.figure6_client_counts
    results = {wl: {} for wl in workloads}

    for wl in workloads:
        print(f"\n{'='*60}")
        print(f"WORKLOAD: {wl.upper()}")
        print(f"{'='*60}")

        # Pre-create table once per workload (updates are negligible
        # relative to table size, so reuse is safe)
        table = IndexTable(config.num_rows, config.entries_per_row)
        prepopulate(table, config, config.prepopulate_fill)

        for nc in client_counts:
            mops_trials = []
            for trial in range(config.num_trials):
                mops = run_simulation(config, wl, nc, shared_table=table)
                mops_trials.append(mops)
            results[wl][nc] = np.mean(mops_trials)

    return results


def print_results(results: dict):
    """Print results table."""
    print(f"\n{'='*70}")
    print("FIGURE 6 RESULTS: Throughput (MOPS) vs Number of Clients")
    print(f"{'='*70}")

    workloads = list(results.keys())
    header = f"{'Clients':>8}"
    for wl in workloads:
        header += f" | {wl.upper():>12}"
    print(header)
    print("-" * len(header))

    all_clients = sorted(set(
        c for wl_results in results.values() for c in wl_results.keys()
    ))
    for nc in all_clients:
        row = f"{nc:>8}"
        for wl in workloads:
            mops = results[wl].get(nc, 0)
            row += f" | {mops:>12.3f}"
        print(row)
    print(f"{'='*70}")


def plot_results(results: dict):
    """Plot Figure 6 style charts."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        titles = {
            "ycsb-c": "(a) Read only (YCSB-C)",
            "ycsb-b": "(b) 95% read, 5% update (YCSB-B)",
            "ycsb-a": "(c) 50% read, 50% update (YCSB-A)"
        }
        # Paper reference throughput from researcher data (Section 5.3)
        paper_data = {
            "ycsb-c": {
                "clients": [10, 20, 40, 80, 160, 320],
                "mops": [2.99, 5.953, 11.494, 22.579, 39.369, 46.493]
            },
            "ycsb-b": {
                "clients": [8, 16, 40, 80, 160, 320],
                "mops": [1.936, 3.764, 8.914, 16.538, 27.736, 38.570]
            },
            "ycsb-a": {
                "clients": [8, 16, 40, 80, 160, 320],
                "mops": [0.909, 1.819, 4.275, 7.860, 13.809, 22.353]
            },
        }

        for idx, wl in enumerate(["ycsb-c", "ycsb-b", "ycsb-a"]):
            ax = axes[idx]
            wl_results = results[wl]
            clients = sorted(wl_results.keys())
            mops = [wl_results[c] for c in clients]

            ax.plot(clients, mops, 'o-', color='#4363d8', linewidth=2,
                    markersize=6, label='RCuckoo (sim)')

            # Plot paper reference
            pd = paper_data[wl]
            ax.plot(pd["clients"], pd["mops"], 's--', color='#e6194B',
                    linewidth=1.5, markersize=5, alpha=0.7,
                    label='RCuckoo (paper)')

            ax.set_xlabel('clients')
            ax.set_ylabel('MOPS')
            ax.set_title(titles[wl])
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.set_xlim(0, max(clients) + 20)
            ax.set_ylim(bottom=0)

        plt.tight_layout()
        out = '/Users/ilank/Documents/Openu/Thesis/RCuckko_Testing/figure6_results.png'
        plt.savefig(out, dpi=150)
        print(f"\nPlot saved to {out}")
    except Exception as e:
        print(f"\nCould not plot: {e}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    # ESTIMATION: Scale table down for simulation feasibility.
    # Paper uses 100M entries; we scale to 10M to keep prepopulation
    # fast while having enough locks (78,125) to avoid artificial
    # Zipf contention. 100M entries would be ideal but too slow.
    SCALE_FACTOR = 10

    config = RCuckooConfig(
        total_entries=100_000_000 // SCALE_FACTOR,  # 10M entries
        entries_per_row=8,         # Paper Section 3.1: 8
        locality_f=2.3,           # Paper Section 3.3: 2.3
        rows_per_lock=16,         # Paper Section 3.4.1: 16
        locks_per_mcas=64,        # Paper Section 3.4.1: 64
        max_search_depth=5,       # Paper Section 3.2.3: 5
        prepopulate_fill=0.9,     # Paper Section 5.3: 90M/100M
        zipf_theta=0.99,          # Paper Section 5.3: 0.99
        sim_duration_us=20_000,   # ESTIMATION: 20ms simulated time
        num_trials=1,
        figure6_client_counts=[8, 16, 40, 80, 160, 320],
    )

    print("RCuckoo Simulator")
    print("Paper: 'Cuckoo for Clients' - Grant & Snoeren, ATC'25\n")
    print("=== CONFIGURATION ===")
    print(f"  Table entries:     {config.total_entries:,} "
          f"(paper: 100M, scaled 1/{SCALE_FACTOR} - ESTIMATION)")
    print(f"  Entries per row:   {config.entries_per_row} (paper: 8)")
    print(f"  Number of rows:    {config.num_rows:,}")
    print(f"  Entry size:        {config.entry_size_bytes}B "
          f"(paper: 8B = 32b key + 32b value)")
    print(f"  Locality f:        {config.locality_f} (paper: 2.3)")
    print(f"  Rows per lock:     {config.rows_per_lock} (paper: 16)")
    print(f"  Total locks:       {config.num_locks:,}")
    print(f"  Locks per MCAS:    {config.locks_per_mcas} (paper: 64)")
    print(f"  Max search depth:  {config.max_search_depth} (paper: 5)")
    print(f"  Client cache:      {config.client_cache_size_bytes//1024}KB "
          f"(paper: 64KB)")
    print(f"  Pre-populate:      {config.prepopulate_fill*100:.0f}% (paper: 90%)")
    print(f"  Zipf theta:        {config.zipf_theta} (paper: 0.99)")
    print(f"  NIC bandwidth:     {config.nic_bandwidth_gbps:.0f} Gbps "
          f"(paper: 100 Gbps, Section 2.2)")
    print(f"  NIC max reads/us:  {config.nic_max_read_ops_per_us} "
          f"(paper Figure 1b: ~75)")
    print(f"  NIC max CAS/us:    {config.nic_max_cas_ops_per_us} "
          f"(paper Figure 1b: ~50)")
    print(f"  RDMA RTT:          {config.rdma_rtt_us} us (ESTIMATION)")
    print(f"  Sim duration:      {config.sim_duration_us/1000:.0f}ms "
          f"(ESTIMATION)")
    print(f"  Client counts:     {config.figure6_client_counts}")
    print()

    results = run_figure6(config)
    print_results(results)
    plot_results(results)


if __name__ == "__main__":
    main()
