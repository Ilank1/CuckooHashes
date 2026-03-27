"""Shared simulation parameters for RCuckoo and OfflineState comparison."""

# ── General ───────────────────────────────────────────────────
SCALE_FACTOR = 10
TOTAL_ENTRIES = 100_000_000 // SCALE_FACTOR  # 10M
ZIPF_THETA = 0.99
RDMA_RTT_US = 1.0
SERVER_RTT_TICKS = 3
SIM_DURATION_US = 20_000
NUM_TRIALS = 1
CLIENT_COUNTS = [8, 16, 40, 80, 160, 320]
ENTRY_SIZE_BYTES = 8
NIC_BANDWIDTH_GBPS = 100.0

# Workloads (YCSB)
WORKLOADS = ["ycsb-c", "ycsb-b", "ycsb-a"]

# Workload sizing
OPS_PER_CLIENT = 50_000
MIN_OPS = 200_000

# ── RCuckoo ───────────────────────────────────────────────────
ENTRIES_PER_ROW = 8
LOCALITY_F = 2.3
ROWS_PER_LOCK = 16
LOCKS_PER_MCAS = 64
MAX_SEARCH_DEPTH = 5
PREPOPULATE_FILL = 0.9
RCUCKOO_CLIENT_CACHE_SIZE_BYTES = 64 * 1024  # 64KB
LOCK_RETRY_LIMIT = 200

# ── OfflineState ──────────────────────────────────────────────
CLIENT_CACHE_SIZE_BYTES = 256 * 1024  # 256KB
SYNC_INTERVAL_TICKS = 100
BLOOM_FILTER_FP_PROB = 0.01
BLOOM_FILTER_AUTO_SCALE = True
NUM_PEER_GROUPS = 16  # GIBF: partition clients into G groups

# ── Sensitivity sweeps (compare.py) ──────────────────────────
SENSITIVITY_NC = 80
ZIPF_THETA_SWEEP = [0.5, 0.7, 0.8, 0.9, 0.95, 0.99]
SYNC_INTERVAL_SWEEP = [10, 50, 100, 500, 1000, 5000]
CACHE_SIZE_SWEEP_KB = [8, 16, 32, 64, 128, 256]

# ── Reference data (RCuckoo ATC'25 Figure 6) ─────────────────
REFERENCE_RCUCKOO = {
    "ycsb-c": {
        "clients": [10, 20, 40, 80, 160, 320],
        "mops": [2.99, 5.953, 11.494, 22.579, 39.369, 46.493],
    },
    "ycsb-b": {
        "clients": [8, 16, 40, 80, 160, 320],
        "mops": [1.936, 3.764, 8.914, 16.538, 27.736, 38.570],
    },
    "ycsb-a": {
        "clients": [8, 16, 40, 80, 160, 320],
        "mops": [0.909, 1.819, 4.275, 7.860, 13.809, 22.353],
    },
}
