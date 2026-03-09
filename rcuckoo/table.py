"""
Remote data structures for the RCuckoo index and lock tables.

Section 3.1 (Grant & Snoeren, ATC'25):
    "a single region of RDMA-registered main memory divided into rows
    of fixed-width entries. Each row contains n associative entries and
    terminates with an 8-bit version number and 64-bit CRC."

Section 3.4:
    "Locks (stored in a bit vector in NIC memory) each protect a tunable
    number (here, two; 16 in our experiments) of index rows."
"""

import numpy as np

from rcuckoo.hashing import EMPTY_KEY


class IndexTable:
    """
    Remote index table backed by numpy arrays.

    Each row holds `entries_per_row` key-value pairs, an 8-bit version
    counter, and a 64-bit CRC (size accounted for, not computed).
    """

    def __init__(self, num_rows: int, entries_per_row: int):
        self.num_rows = num_rows
        self.entries_per_row = entries_per_row
        self.keys = np.full((num_rows, entries_per_row), EMPTY_KEY, dtype=np.uint32)
        self.values = np.zeros((num_rows, entries_per_row), dtype=np.uint32)
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
        """Increment 8-bit row version (Section 3.1)."""
        self.versions[row] = (self.versions[row] + 1) & 0xFF


class LockTable:
    """
    Lock table in NIC device memory (Section 3.4).

    In simulated time, locks are a simple boolean array. Contention is
    modeled by checking lock state each tick.
    """

    def __init__(self, num_locks: int):
        self.num_locks = num_locks
        self.locked = np.zeros(num_locks, dtype=np.bool_)
        self.holder = np.full(num_locks, -1, dtype=np.int32)

    def row_to_lock(self, row_idx: int, rows_per_lock: int) -> int:
        return row_idx // rows_per_lock

    def try_acquire_mcas(self, lock_indices: list, client_id: int) -> bool:
        """
        Simulate MCAS lock acquisition (Section 3.4.1).

        "leverages masked CAS (MCAS) operations to obtain up to 64
        locks simultaneously while avoiding false sharing."
        "To avoid deadlock RCuckoo acquires locks in increasing order."
        """
        indices = sorted(set(idx % self.num_locks for idx in lock_indices))
        for idx in indices:
            if self.locked[idx]:
                return False
        for idx in indices:
            self.locked[idx] = True
            self.holder[idx] = client_id
        return True

    def release_all(self, lock_indices: list):
        for idx in set(lock_indices):
            real_idx = idx % self.num_locks
            self.locked[real_idx] = False
            self.holder[real_idx] = -1
