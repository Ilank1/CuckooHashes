"""
OfflineState client with LRU cache and bloom filters.

Each client maintains:
- An LRU cache (OrderedDict) for key-value pairs
- A local bloom filter tracking keys in its own cache
- A peer bloom filter (OR of all other clients' local filters)
"""

from collections import OrderedDict

from offline_state.bloom_filter import BloomFilter

# Client phases
PHASE_IDLE = 0
PHASE_PEER_READ = 1        # broadcast read to all peers (1 RTT)
PHASE_SERVER_READ = 2      # one-sided RDMA read from server (1 RTT)
PHASE_SERVER_WRITE = 3     # one-sided RDMA write to server (1 RTT)


class OfflineClient:
    __slots__ = (
        'client_id', 'phase', 'op_key', 'op_type',
        'cache', 'max_cache_entries',
        'local_bloom', 'peer_bloom',
        'ops_completed', 'local_hits', 'peer_hits', 'server_hits',
        'rdma_calls', 'ticks_remaining',
    )

    def __init__(self, client_id: int, max_cache_entries: int,
                 bloom_size_bits: int, bloom_num_hashes: int):
        self.client_id = client_id
        self.phase = PHASE_IDLE
        self.op_key = 0
        self.op_type = 'read'

        self.cache = OrderedDict()
        self.max_cache_entries = max_cache_entries

        self.local_bloom = BloomFilter(bloom_size_bits, bloom_num_hashes)
        self.peer_bloom = BloomFilter(bloom_size_bits, bloom_num_hashes)

        self.ops_completed = 0
        self.local_hits = 0
        self.peer_hits = 0
        self.server_hits = 0
        self.rdma_calls = 0
        self.ticks_remaining = 0

    def cache_get(self, key: int):
        """LRU lookup. Returns value if present, else None."""
        if key in self.cache:
            self.cache.move_to_end(key)
            return self.cache[key]
        return None

    def cache_put(self, key: int, value: int):
        """LRU insert. Evicts oldest entry if at capacity."""
        if key in self.cache:
            self.cache.move_to_end(key)
            self.cache[key] = value
            return
        if len(self.cache) >= self.max_cache_entries:
            self.cache.popitem(last=False)  # evict LRU
        self.cache[key] = value
        self.local_bloom.add(key)
