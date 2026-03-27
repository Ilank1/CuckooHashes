"""
Configuration for the OfflineState distributed cache simulator.

OfflineState uses client-side LRU caches with bloom filters and a
periodic offline sync worker that merges bloom filter state across clients.
"""

import math
from dataclasses import dataclass, field

from sim_config import (
    TOTAL_ENTRIES, ZIPF_THETA, ENTRY_SIZE_BYTES,
    CLIENT_CACHE_SIZE_BYTES, BLOOM_FILTER_FP_PROB, BLOOM_FILTER_AUTO_SCALE,
    SERVER_RTT_TICKS, SYNC_INTERVAL_TICKS, RDMA_RTT_US, NIC_BANDWIDTH_GBPS,
    SIM_DURATION_US, NUM_TRIALS, CLIENT_COUNTS, NUM_PEER_GROUPS,
)


@dataclass
class OfflineStateConfig:
    # Table / workload params (same as RCuckoo for fair comparison)
    total_entries: int = TOTAL_ENTRIES
    zipf_theta: float = ZIPF_THETA
    entry_size_bytes: int = ENTRY_SIZE_BYTES

    # Client cache
    client_cache_size_bytes: int = CLIENT_CACHE_SIZE_BYTES

    # Bloom filter params — sized via optimal formula:
    #   m = -(n * ln(p)) / (ln(2)^2),  k = (m/n) * ln(2)
    bloom_filter_fp_prob: float = BLOOM_FILTER_FP_PROB
    bloom_filter_auto_scale: bool = BLOOM_FILTER_AUTO_SCALE

    # Latency model
    local_cache_rtt: int = 0
    peer_read_rtt: int = 1       # broadcast to all peers (same rack, 1 tick)
    server_rtt_ticks: int = SERVER_RTT_TICKS

    # GIBF: number of client groups for group-indexed bloom filter
    num_peer_groups: int = NUM_PEER_GROUPS

    # Offline worker sync interval (in ticks)
    sync_interval_ticks: int = SYNC_INTERVAL_TICKS

    # NIC + timing (same as RCuckoo)
    rdma_rtt_us: float = RDMA_RTT_US
    nic_bandwidth_gbps: float = NIC_BANDWIDTH_GBPS
    read_threshold_bytes: int = 256

    # Simulation
    sim_duration_us: int = SIM_DURATION_US
    num_trials: int = NUM_TRIALS
    figure6_client_counts: list = field(
        default_factory=lambda: list(CLIENT_COUNTS)
    )

    def effective_bloom_params(self, num_clients: int) -> tuple:
        """Compute optimal bloom filter (size_bits, num_hashes) from formula.

        Uses m = -(n * ln(p)) / (ln(2)^2), k = (m/n) * ln(2).
        """
        n = self.max_cache_entries
        if self.bloom_filter_auto_scale:
            scale = max(1, int(math.ceil(num_clients ** 0.5)))
            n = n * scale

        p = self.bloom_filter_fp_prob
        m = int(-(n * math.log(p)) / (math.log(2) ** 2))
        m = min(m, self.total_entries * 10)  # cap at reasonable max
        k = max(1, int((m / n) * math.log(2)))
        return m, k

    def gibf_params(self, num_clients: int) -> tuple:
        """Compute GIBF (size_positions, num_hashes) for per-group sizing.

        Size for the items per group: (num_clients / num_groups) * cache_entries.
        With Zipf overlap, distinct items per group ≈ cache_entries * small_factor.
        """
        num_groups = min(self.num_peer_groups, num_clients)
        group_size = max(1, num_clients // num_groups)
        # Each group has group_size clients contributing to the same bloom positions.
        # Scale n by group_size to keep per-group FPR at the target.
        n = self.max_cache_entries * group_size
        p = self.bloom_filter_fp_prob
        m = int(-(n * math.log(p)) / (math.log(2) ** 2))
        m = min(m, self.total_entries * 10)
        k = max(1, int((m / n) * math.log(2)))
        return m, k

    @property
    def max_cache_entries(self) -> int:
        return self.client_cache_size_bytes // self.entry_size_bytes

    @property
    def nic_bandwidth_bytes_per_us(self) -> float:
        return self.nic_bandwidth_gbps * 1e9 / 8 / 1e6
