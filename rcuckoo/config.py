"""
Configuration for the RCuckoo simulator.

All parameters reference Grant & Snoeren, ATC'25:
"Cuckoo for Clients: Disaggregated Cuckoo Hashing"

Parameters marked ESTIMATION are derived from published figures or calibrated
to match reported throughput; all others are stated explicitly in the text.
"""

from dataclasses import dataclass, field


@dataclass
class RCuckooConfig:
    # --- Table parameters (Section 3.1, Section 5.3) ---
    # "100-M-entry table" with "8 entries per row"
    total_entries: int = 100_000_000
    entries_per_row: int = 8  # Section 3.1: "we use 8 in practice"

    # Section 5.3: "32-bit key and 32-bit value"
    key_size_bits: int = 32
    value_size_bits: int = 32

    # --- Hashing (Section 3.3) ---
    # "we set f = 2.3 based on this empirical data"
    locality_f: float = 2.3
    # "We use xxHash in our implementation" - we use integer mixing
    hash_salt_1: int = 0xDEADBEEF
    hash_salt_2: int = 0xCAFEBABE
    hash_salt_3: int = 0x12345678

    # --- Locking (Section 3.4) ---
    # "16 rows per lock" (Section 3.4.1)
    rows_per_lock: int = 16
    # "single-bit locks correspond to 64 locks per message"
    locks_per_mcas: int = 64

    # --- Search (Section 3.2.3) ---
    # "maximum search depth (we use a depth of five)"
    max_search_depth: int = 5

    # --- Client cache (Section 3.2.3, Section 5.1) ---
    # "64-KB client index-table caches"
    client_cache_size_bytes: int = 64 * 1024

    # --- Pre-population (Section 5.3) ---
    # "pre-populate it with 90 M entries" => 90% fill
    prepopulate_fill: float = 0.9

    # --- Workload (Section 5.3) ---
    # "Zipf(0.99) distribution"
    zipf_theta: float = 0.99

    # --- RDMA timing (Section 1, Section 2.2) ---
    # "RDMA latency is approximately 1 us" (NIC processing time).
    # ESTIMATION: Effective per-operation RTT including network traversal,
    # doorbell posting, and CQ polling is ~3us in practice (derived from
    # Figure 6 YCSB-C: 10 clients -> 2.99 MOPS -> ~3.3us/op/client).
    rdma_rtt_us: float = 3.0

    # --- NIC capacity (Section 2.2, Figures 1a-c) ---
    # "100-Gbps ConnectX-5 testbed hardware" (Section 2.2)
    nic_bandwidth_gbps: float = 100.0
    # Figure 1b: reads/writes scale to ~75 MOPS, CAS ~50 MOPS (device memory)
    # Figure 1c: contended CAS on device memory ~16 MOPS (single address ~3 MOPS)
    nic_max_read_ops_per_us: int = 75
    nic_max_cas_ops_per_us: int = 50  # device memory, independent addresses
    # Section 3.2.1: covering read threshold
    # Researcher notes: read_threshold_bytes = 256
    read_threshold_bytes: int = 256

    # --- Figure 6 client counts (from Figure 6 x-axis) ---
    # Validated against researcher data: [8, 16, 40, 80, 160, 320]
    figure6_client_counts: list = field(
        default_factory=lambda: [8, 16, 40, 80, 160, 320]
    )

    # --- Simulation ---
    # ESTIMATION: simulation duration in simulated microseconds
    sim_duration_us: int = 100_000  # 100ms of simulated time
    num_trials: int = 1

    @property
    def num_rows(self) -> int:
        return self.total_entries // self.entries_per_row

    @property
    def entry_size_bytes(self) -> int:
        return (self.key_size_bits + self.value_size_bits) // 8

    @property
    def row_size_bytes(self) -> int:
        # entries + 1-byte version + 8-byte CRC (Section 3.1)
        return self.entries_per_row * self.entry_size_bytes + 1 + 8

    @property
    def num_locks(self) -> int:
        return (self.num_rows + self.rows_per_lock - 1) // self.rows_per_lock

    @property
    def cache_rows(self) -> int:
        return self.client_cache_size_bytes // self.row_size_bytes

    @property
    def nic_bandwidth_bytes_per_us(self) -> float:
        """NIC bandwidth in bytes per microsecond."""
        return self.nic_bandwidth_gbps * 1e9 / 8 / 1e6  # 100 Gbps = 12500 B/us
