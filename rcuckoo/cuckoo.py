"""Cuckoo path search and execution using BFS."""

from collections import deque
from typing import Optional

import numpy as np

from rcuckoo.config import RCuckooConfig
from rcuckoo.hashing import EMPTY_KEY, compute_locations
from rcuckoo.table import IndexTable


def group_locks_for_mcas(lock_indices: list, locks_per_mcas: int) -> list:
    """Group locks into MCAS groups (each group fits in one 64-bit CAS)."""
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
    BFS search for a cuckoo eviction path within locked rows.

    Returns a list of moves or None if no path found within max_search_depth.
    """
    queue = deque()
    visited = set()
    parent = {}

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

        eL1, eL2 = compute_locations(
            entry_key, config.num_rows, config.locality_f,
            config.hash_salt_1, config.hash_salt_2, config.hash_salt_3
        )
        alt_row = eL2 if row_idx == eL1 else eL1

        if alt_row not in locked_rows:
            continue

        empty_slot = table.empty_slot_in_row(alt_row)
        if empty_slot >= 0:
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

        for s in range(config.entries_per_row):
            if (alt_row, s) not in visited and table.keys[alt_row, s] != EMPTY_KEY:
                visited.add((alt_row, s))
                parent[(alt_row, s)] = (row_idx, slot_idx, alt_row, s)
                queue.append((alt_row, s, depth + 1))

    return None


def execute_cuckoo_path(table: IndexTable, path: list):
    """Execute a cuckoo eviction path in reverse order."""
    for move in reversed(path):
        fr, fs = move['from_row'], move['from_slot']
        tr, ts = move['to_row'], move['to_slot']
        table.keys[tr, ts] = move['key']
        table.values[tr, ts] = move['value']
        table.increment_version(tr)
        table.keys[fr, fs] = EMPTY_KEY
        table.values[fr, fs] = 0
        table.increment_version(fr)
