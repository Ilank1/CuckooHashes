"""
Bloom filter for OfflineState distributed cache.

Each client maintains a local bloom filter tracking its cached keys,
and a peer bloom filter (bitwise OR of all other clients' local filters)
used to determine whether a broadcast peer read is worthwhile.
"""

import numpy as np


class BloomFilter:
    """Numpy-backed bloom filter with integer hashing."""

    def __init__(self, size_bits: int = 81920, num_hashes: int = 7):
        self.size_bits = size_bits
        self.num_hashes = num_hashes
        self.bits = np.zeros(size_bits, dtype=np.uint8)
        # Seeds for each hash function
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
        return h % self.size_bits

    def add(self, key: int):
        for seed in self._seeds:
            idx = self._hash(key, int(seed))
            self.bits[idx] = 1

    def contains(self, key: int) -> bool:
        for seed in self._seeds:
            idx = self._hash(key, int(seed))
            if self.bits[idx] == 0:
                return False
        return True

    def clear(self):
        self.bits[:] = 0

    def fill_rate(self) -> float:
        return float(np.sum(self.bits)) / self.size_bits

    def rebuild_from_keys(self, keys):
        """Clear and re-add all keys (handles evictions from LRU)."""
        self.clear()
        for k in keys:
            self.add(k)

    def copy(self):
        bf = BloomFilter(self.size_bits, self.num_hashes)
        bf.bits[:] = self.bits
        bf._seeds = self._seeds
        return bf

    @staticmethod
    def bitwise_or(filters):
        """Compute bitwise OR of multiple bloom filters. Returns a new filter."""
        if not filters:
            raise ValueError("Need at least one filter")
        result = filters[0].copy()
        for f in filters[1:]:
            np.bitwise_or(result.bits, f.bits, out=result.bits)
        return result
