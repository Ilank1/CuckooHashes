"""
Client state machine for RCuckoo operations (Section 3.2).

Each phase corresponds to one RDMA round trip:
- Read: 1 RTT (Section 3.2.1)
- Update: 2 RTTs minimum (Section 3.2.2)
- Insert: 2+ RTTs (Section 3.2.3)
"""

from collections import deque

# Operation phases - each consumes 1 RDMA round trip
PHASE_IDLE = 0
PHASE_READ_ISSUED = 1      # Read: RDMA read both rows (1 RTT)
PHASE_UPDATE_LOCK = 2      # Update: try acquire locks (1 RTT)
PHASE_UPDATE_WRITE = 4     # Update: write + release (1 RTT)
PHASE_INSERT_LOCK = 5      # Insert: try acquire locks (1 RTT)
PHASE_INSERT_SEARCH = 6    # Insert: search within locked rows
PHASE_INSERT_WRITE = 7     # Insert: execute cuckoo path + release


class ClientState:
    """State for one simulated client."""

    __slots__ = [
        'client_id', 'phase', 'op_type', 'op_key', 'op_value',
        'L1', 'L2', 'lock_indices', 'acquired_locks',
        'mcas_groups', 'mcas_group_idx',
        'ops_completed', 'lock_retries',
        'cache', 'cache_order', 'max_cache_rows',
        'locked_rows'
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
        self.locked_rows = set()

    def cache_row(self, row_idx: int, keys, values):
        """Cache a row's data with LRU eviction (Section 3.2.3, 5.1: 64KB cache)."""
        if row_idx in self.cache:
            self.cache[row_idx] = (keys.copy(), values.copy())
            return
        if len(self.cache) >= self.max_cache_rows:
            evict = self.cache_order.popleft()
            self.cache.pop(evict, None)
        self.cache[row_idx] = (keys.copy(), values.copy())
        self.cache_order.append(row_idx)
