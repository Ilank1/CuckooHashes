"""
Client state machine for RCuckoo operations.

Each non-IDLE phase corresponds to one RDMA round trip:
- Read: 1 RTT
- Update: 2 RTTs minimum (lock + write)
- Insert: 2+ RTTs (lock + search + write)
"""

from collections import deque

# Operation phases — each consumes 1 RDMA round trip
PHASE_IDLE = 0
PHASE_READ_ISSUED = 1
PHASE_UPDATE_LOCK = 2
PHASE_UPDATE_WRITE = 4
PHASE_INSERT_LOCK = 5
PHASE_INSERT_SEARCH = 6   # no RDMA — uses cached data
PHASE_INSERT_WRITE = 7


class ClientState:
    __slots__ = [
        'client_id', 'phase', 'op_type', 'op_key', 'op_value',
        'L1', 'L2', 'lock_indices', 'acquired_locks',
        'mcas_groups', 'mcas_group_idx',
        'ops_completed', 'lock_retries',
        'cache', 'cache_order', 'max_cache_rows',
        'locked_rows', 'rdma_calls', 'ticks_remaining',
    ]

    def __init__(self, client_id: int, max_cache_rows: int):
        self.client_id = client_id
        self.phase = PHASE_IDLE
        self.op_type = None
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
        self.locked_rows = set()
        self.rdma_calls = 0
        self.ticks_remaining = 0

    def cache_row(self, row_idx: int, keys, values):
        if row_idx in self.cache:
            self.cache[row_idx] = (keys.copy(), values.copy())
            return
        if len(self.cache) >= self.max_cache_rows:
            evict = self.cache_order.popleft()
            self.cache.pop(evict, None)
        self.cache[row_idx] = (keys.copy(), values.copy())
        self.cache_order.append(row_idx)
