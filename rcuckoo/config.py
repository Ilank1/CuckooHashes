"""Configuration for the RCuckoo simulator."""

from dataclasses import dataclass, field

from sim_config import (
    TOTAL_ENTRIES, ENTRIES_PER_ROW, LOCALITY_F, ROWS_PER_LOCK,
    LOCKS_PER_MCAS, MAX_SEARCH_DEPTH, RCUCKOO_CLIENT_CACHE_SIZE_BYTES,
    PREPOPULATE_FILL, ZIPF_THETA, RDMA_RTT_US, SERVER_RTT_TICKS,
    NIC_BANDWIDTH_GBPS, CLIENT_COUNTS, SIM_DURATION_US, NUM_TRIALS,
)


@dataclass
class RCuckooConfig:
    # Table parameters
    total_entries: int = TOTAL_ENTRIES
    entries_per_row: int = ENTRIES_PER_ROW
    key_size_bits: int = 32
    value_size_bits: int = 32

    # Dependent hashing
    locality_f: float = LOCALITY_F
    hash_salt_1: int = 0xDEADBEEF
    hash_salt_2: int = 0xCAFEBABE
    hash_salt_3: int = 0x12345678

    # Locking
    rows_per_lock: int = ROWS_PER_LOCK
    locks_per_mcas: int = LOCKS_PER_MCAS

    # Cuckoo search
    max_search_depth: int = MAX_SEARCH_DEPTH

    # Client cache
    client_cache_size_bytes: int = RCUCKOO_CLIENT_CACHE_SIZE_BYTES

    # Pre-population
    prepopulate_fill: float = PREPOPULATE_FILL

    # Workload
    zipf_theta: float = ZIPF_THETA

    # RDMA timing
    rdma_rtt_us: float = RDMA_RTT_US
    server_rtt_ticks: int = SERVER_RTT_TICKS

    # NIC capacity
    nic_bandwidth_gbps: float = NIC_BANDWIDTH_GBPS
    nic_max_read_ops_per_us: int = 75
    nic_max_cas_ops_per_us: int = 50
    read_threshold_bytes: int = 256

    # Evaluation
    figure6_client_counts: list = field(
        default_factory=lambda: list(CLIENT_COUNTS)
    )

    # Simulation
    sim_duration_us: int = SIM_DURATION_US
    num_trials: int = NUM_TRIALS

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
