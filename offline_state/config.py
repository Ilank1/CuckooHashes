"""
Configuration for the OfflineState distributed cache simulator.

OfflineState uses client-side LRU caches with bloom filters and a
periodic offline sync worker that merges bloom filter state across clients.
"""

from dataclasses import dataclass, field


@dataclass
class OfflineStateConfig:
    # Table / workload params (same as RCuckoo for fair comparison)
    total_entries: int = 10_000_000
    zipf_theta: float = 0.99
    entry_size_bytes: int = 8  # 32-bit key + 32-bit value

    # Client cache
    client_cache_size_bytes: int = 64 * 1024  # 64KB per client

    # Bloom filter params
    # ~10KB (81920 bits), 7 hashes => ~1% FPR for 8192 items
    bloom_filter_size_bits: int = 81920
    bloom_filter_num_hashes: int = 7

    # Latency model
    local_cache_rtt: int = 0
    peer_read_rtt: int = 1       # broadcast to all peers (same rack, 1 tick)
    server_rtt_ticks: int = 3    # server access cost in ticks (cross-rack)

    # Offline worker sync interval (in ticks)
    sync_interval_ticks: int = 100

    # NIC + timing (same as RCuckoo)
    rdma_rtt_us: float = 1.0
    nic_bandwidth_gbps: float = 100.0
    read_threshold_bytes: int = 256

    # Simulation
    sim_duration_us: int = 20_000
    num_trials: int = 1
    figure6_client_counts: list = field(
        default_factory=lambda: [8, 16, 40, 80, 160, 320]
    )

    @property
    def max_cache_entries(self) -> int:
        return self.client_cache_size_bytes // self.entry_size_bytes

    @property
    def nic_bandwidth_bytes_per_us(self) -> float:
        return self.nic_bandwidth_gbps * 1e9 / 8 / 1e6
