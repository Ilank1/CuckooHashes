"""
Entry point for running the OfflineState simulator standalone.

Usage:
    python -m offline_state
"""

from offline_state.config import OfflineStateConfig
from offline_state.engine import run_simulation


def main():
    config = OfflineStateConfig(
        total_entries=10_000_000,
        zipf_theta=0.99,
        client_cache_size_bytes=64 * 1024,
        entry_size_bytes=8,
        sync_interval_ticks=100,
        sim_duration_us=20_000,
        num_trials=1,
        figure6_client_counts=[8, 16, 40, 80, 160, 320],
    )

    print("OfflineState Simulator")
    print("======================\n")
    print(f"  Total entries:       {config.total_entries:,}")
    print(f"  Zipf theta:          {config.zipf_theta}")
    print(f"  Client cache:        {config.client_cache_size_bytes // 1024}KB "
          f"({config.max_cache_entries} entries)")
    print(f"  Bloom filter:        {config.bloom_filter_size_bits} bits, "
          f"{config.bloom_filter_num_hashes} hashes")
    print(f"  Sync interval:       {config.sync_interval_ticks} ticks")
    print(f"  RDMA RTT:            {config.rdma_rtt_us} us")
    print(f"  NIC bandwidth:       {config.nic_bandwidth_gbps} Gbps")
    print(f"  Sim duration:        {config.sim_duration_us / 1000:.0f}ms")
    print(f"  Client counts:       {config.figure6_client_counts}")
    print()

    workloads = ["ycsb-c", "ycsb-b", "ycsb-a"]

    for wl in workloads:
        print(f"\n{'='*60}")
        print(f"WORKLOAD: {wl.upper()}")
        print(f"{'='*60}")
        for nc in config.figure6_client_counts:
            mops, stats = run_simulation(config, wl, nc)


if __name__ == "__main__":
    main()
