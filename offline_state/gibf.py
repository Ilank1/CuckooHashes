"""
Group-Indexed Bloom Filter (GIBF) for OfflineState distributed cache.

Instead of a single bit per position (standard bloom), each position stores
a G-bit group bitmap indicating which client group(s) set that bit.

Lookup: hash key to k positions, AND the bitmaps → candidate group set.
This tells which groups likely have the key
"""

import numpy as np


class GroupIndexedBloomFilter:
    """Bitmap-based bloom filter that tracks per-group membership."""

    def __init__(self, size_positions: int, num_hashes: int, num_groups: int):
        self.size = size_positions
        self.num_hashes = num_hashes
        self.num_groups = num_groups
        # Each position stores a uint32 bitmap (supports up to 32 groups)
        self.bitmaps = np.zeros(size_positions, dtype=np.uint32)
        # Same seeds as BloomFilter for consistency
        self._seeds = np.array(
            [0xDEADBEEF + i * 0x9E3779B9 for i in range(num_hashes)],
            dtype=np.uint64
        )

    def _hash(self, key: int, seed: int) -> int:
        """Integer mixing hash (murmurhash-style finalizer)."""
        h = (key ^ seed) & 0xFFFFFFFFFFFFFFFF
        h = ((h ^ (h >> 33)) * 0xFF51AFD7ED558CCD) & 0xFFFFFFFFFFFFFFFF
        h = ((h ^ (h >> 33)) * 0xC4CEB9FE1A85EC53) & 0xFFFFFFFFFFFFFFFF
        h = h ^ (h >> 33)
        return h % self.size

    def add(self, key: int, group_id: int):
        """Set group_id bit at all k hash positions for this key."""
        bit = np.uint32(1 << group_id)
        for seed in self._seeds:
            idx = self._hash(key, int(seed))
            self.bitmaps[idx] |= bit

    def query(self, key: int) -> set:
        """Return set of candidate group IDs (AND of bitmaps at k positions)."""
        result = np.uint32((1 << self.num_groups) - 1)  # all bits set
        for seed in self._seeds:
            idx = self._hash(key, int(seed))
            result &= self.bitmaps[idx]
        if result == 0:
            return set()
        return {g for g in range(self.num_groups) if result & (1 << g)}

    def query_excluding(self, key: int, exclude_group: int) -> set:
        """Query and exclude own group in one pass."""
        mask = np.uint32(((1 << self.num_groups) - 1) & ~(1 << exclude_group))
        result = mask
        for seed in self._seeds:
            idx = self._hash(key, int(seed))
            result &= self.bitmaps[idx]
        if result == 0:
            return set()
        return {g for g in range(self.num_groups) if result & (1 << g)}

    def clear(self):
        self.bitmaps[:] = 0

    def fill_rate(self) -> float:
        """Fraction of positions with any group bit set."""
        return float(np.count_nonzero(self.bitmaps)) / self.size
