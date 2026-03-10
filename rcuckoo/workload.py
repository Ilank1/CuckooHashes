"""Workload generation and table pre-population."""

import time

import numpy as np

from rcuckoo.config import RCuckooConfig
from rcuckoo.hashing import EMPTY_KEY, compute_locations_batch
from rcuckoo.table import IndexTable


def ycsb_zipf_keys(num_keys: int, theta: float, size: int) -> np.ndarray:
    """
    Generate YCSB-compatible bounded Zipfian key indices over [0, num_keys).

    Implements the YCSB ZipfianGenerator algorithm:
        P(rank k) ~ 1/(k+1)^theta for k=0..N-1
    """
    N = num_keys
    if abs(theta - 1.0) < 1e-6:
        theta = 0.999

    alpha = 1.0 / (1.0 - theta)

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
    Generate a YCSB workload.

    YCSB-C: 100% read, YCSB-B: 95% read / 5% update, YCSB-A: 50/50.
    """
    if workload_type == "ycsb-c":
        read_ratio = 1.0
    elif workload_type == "ycsb-b":
        read_ratio = 0.95
    elif workload_type == "ycsb-a":
        read_ratio = 0.50
    else:
        raise ValueError(f"Unknown workload: {workload_type}")

    key_indices = ycsb_zipf_keys(num_keys, zipf_theta, num_ops)
    key_samples = key_indices.astype(np.uint32)
    is_read = np.random.random(num_ops) < read_ratio

    return key_samples, is_read


def prepopulate(table: IndexTable, config: RCuckooConfig, fill_fraction: float):
    """Pre-populate the index table to the given fill fraction."""
    target = int(config.total_entries * fill_fraction)
    print(f"Pre-populating table: {target:,} entries "
          f"({fill_fraction*100:.0f}% of {config.total_entries:,})...")

    row_fill = np.zeros(config.num_rows, dtype=np.int32)
    inserted = 0
    batch_size = 500_000
    key_start = 0
    t0 = time.monotonic()

    while inserted < target:
        batch_end = min(key_start + batch_size, 0xFFFFFFFF)
        keys = np.arange(key_start, batch_end, dtype=np.uint32)
        keys = keys[keys != EMPTY_KEY]

        L1s, L2s = compute_locations_batch(
            keys, config.num_rows, config.locality_f,
            config.hash_salt_1, config.hash_salt_2, config.hash_salt_3
        )

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
