"""
Entry point for running the OfflineState simulator standalone.

Usage:
    python -m offline_state
"""

from offline_state.config import OfflineStateConfig
from offline_state.engine import run_simulation
from rcuckoo.workload import generate_workload
from sim_config import (
    TOTAL_ENTRIES, ZIPF_THETA, ENTRY_SIZE_BYTES,
    CLIENT_CACHE_SIZE_BYTES, SYNC_INTERVAL_TICKS,
    RDMA_RTT_US, SERVER_RTT_TICKS, SIM_DURATION_US,
    NUM_TRIALS, CLIENT_COUNTS, OPS_PER_CLIENT, MIN_OPS,
    WORKLOADS,
)


def main():
    config = OfflineStateConfig(
        total_entries=TOTAL_ENTRIES,
        zipf_theta=ZIPF_THETA,
        client_cache_size_bytes=CLIENT_CACHE_SIZE_BYTES,
        entry_size_bytes=ENTRY_SIZE_BYTES,
        sync_interval_ticks=SYNC_INTERVAL_TICKS,
        rdma_rtt_us=RDMA_RTT_US,
        server_rtt_ticks=SERVER_RTT_TICKS,
        sim_duration_us=SIM_DURATION_US,
        num_trials=NUM_TRIALS,
        figure6_client_counts=CLIENT_COUNTS,
    )

    print("OfflineState Simulator")
    print("======================\n")
    print(f"  Total entries:       {config.total_entries:,}")
    print(f"  Zipf theta:          {config.zipf_theta}")
    print(f"  Client cache:        {config.client_cache_size_bytes // 1024}KB "
          f"({config.max_cache_entries} entries)")
    m, k = config.effective_bloom_params(8)  # example for 8 clients
    print(f"  Bloom filter:        FPR={config.bloom_filter_fp_prob}, "
          f"auto_scale={config.bloom_filter_auto_scale} "
          f"(e.g. 8 clients: {m} bits, {k} hashes)")
    print(f"  Sync interval:       {config.sync_interval_ticks} ticks")
    print(f"  RDMA RTT:            {config.rdma_rtt_us} us")
    print(f"  NIC bandwidth:       {config.nic_bandwidth_gbps} Gbps")
    print(f"  Sim duration:        {config.sim_duration_us / 1000:.0f}ms")
    print(f"  Client counts:       {config.figure6_client_counts}")
    print()

    workloads = WORKLOADS

    # Pre-generate workloads
    max_ops = max(c * OPS_PER_CLIENT for c in CLIENT_COUNTS)
    max_ops = max(max_ops, MIN_OPS)
    pregenerated = {}
    for wl in workloads:
        key_samples, is_read = generate_workload(
            wl, config.total_entries, config.zipf_theta, max_ops)
        pregenerated[wl] = (key_samples, is_read)

    for wl in workloads:
        print(f"\n{'='*60}")
        print(f"WORKLOAD: {wl.upper()}")
        print(f"{'='*60}")
        for nc in config.figure6_client_counts:
            mops, stats = run_simulation(
                config, wl, nc, pregenerated_workload=pregenerated[wl])


if __name__ == "__main__":
    main()
