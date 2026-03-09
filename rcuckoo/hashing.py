"""
Hash functions for dependent cuckoo hashing (Section 3.3).

Grant & Snoeren, ATC'25:
    L1(K) = h1(K) mod T
    L2(K) = (L1 + (h2(K) mod floor(f^(f + Z(h3(K)))))) mod T

T = number of rows, Z(x) = trailing zeros of x, f = locality parameter.
The original uses xxHash; we substitute a fast integer mixing function.
"""

import numpy as np

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
    Compute two cuckoo hash locations for a key.

    Section 3.3:
        L1(K) = h1(K) mod T
        L2(K) = (L1 + (h2(K) mod floor(f^(f + Z(h3(K)))))) mod T
    """
    h1 = _hash_with_salt(key, salt1)
    h2 = _hash_with_salt(key, salt2)
    h3 = _hash_with_salt(key, salt3)

    L1 = h1 % num_rows

    if h3 == 0:
        z = 32
    else:
        z = (h3 & (-h3)).bit_length() - 1

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


def compute_locations_batch(keys: np.ndarray, num_rows: int, f: float,
                            salt1: int, salt2: int, salt3: int):
    """Batch compute locations for a numpy array of keys. Fully vectorized."""
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

    # Vectorized trailing zeros: lsb = x & (-x), ctz = log2(lsb)
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
