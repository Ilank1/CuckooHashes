"""Configuration for the RCuckoo simulator."""

from dataclasses import dataclass, field


@dataclass
class RCuckooConfig:
    # Table parameters
    total_entries: int = 100_000_000
    entries_per_row: int = 8
    key_size_bits: int = 32
    value_size_bits: int = 32

    # Dependent hashing
    locality_f: float = 2.3
    hash_salt_1: int = 0xDEADBEEF
    hash_salt_2: int = 0xCAFEBABE
    hash_salt_3: int = 0x12345678

    # Locking
    rows_per_lock: int = 16
    locks_per_mcas: int = 64

    # Cuckoo search
    max_search_depth: int = 5

    # Client cache
    client_cache_size_bytes: int = 64 * 1024

    # Pre-population
    prepopulate_fill: float = 0.9

    # Workload
    zipf_theta: float = 0.99

    # RDMA timing
    rdma_rtt_us: float = 1.0
    server_rtt_ticks: int = 3  # server access cost in ticks (cross-rack)

    # NIC capacity
    nic_bandwidth_gbps: float = 100.0
    nic_max_read_ops_per_us: int = 75
    nic_max_cas_ops_per_us: int = 50
    read_threshold_bytes: int = 256

    # Evaluation
    figure6_client_counts: list = field(
        default_factory=lambda: [8, 16, 40, 80, 160, 320]
    )

    # Simulation
    sim_duration_us: int = 100_000
    num_trials: int = 1

    @property
    def num_rows(self) -> int:
        return self.total_entries // self.entries_per_row

    @property
    def entry_size_bytes(self) -> int:
        return (self.key_size_bits + self.value_size_bits) // 8

    @property
    def row_size_bytes(self) -> int:
        return self.entries_per_row * self.entry_size_bytes + 1 + 8

    @property
    def num_locks(self) -> int:
        return (self.num_rows + self.rows_per_lock - 1) // self.rows_per_lock

    @property
    def cache_rows(self) -> int:
        return self.client_cache_size_bytes // self.row_size_bytes

    @property
    def nic_bandwidth_bytes_per_us(self) -> float:
        return self.nic_bandwidth_gbps * 1e9 / 8 / 1e6
