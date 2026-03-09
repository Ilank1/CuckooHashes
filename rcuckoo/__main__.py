"""
Entry point for running the RCuckoo simulator.

Usage:
    python -m rcuckoo
"""

from rcuckoo.config import RCuckooConfig
from rcuckoo.evaluation import run_figure6, print_results, plot_results


def main():
    # ESTIMATION: Scale table down for simulation feasibility.
    # Section 5.3 uses 100M entries; we scale to 10M to keep prepopulation
    # fast while preserving enough locks (78,125) for realistic contention.
    SCALE_FACTOR = 10

    config = RCuckooConfig(
        total_entries=100_000_000 // SCALE_FACTOR,
        entries_per_row=8,         # Section 3.1: 8
        locality_f=2.3,           # Section 3.3: 2.3
        rows_per_lock=16,         # Section 3.4.1: 16
        locks_per_mcas=64,        # Section 3.4.1: 64
        max_search_depth=5,       # Section 3.2.3: 5
        prepopulate_fill=0.9,     # Section 5.3: 90M/100M
        zipf_theta=0.99,          # Section 5.3: 0.99
        sim_duration_us=20_000,   # ESTIMATION: 20ms simulated time
        num_trials=1,
        figure6_client_counts=[8, 16, 40, 80, 160, 320],
    )

    print("RCuckoo Simulator")
    print("'Cuckoo for Clients' - Grant & Snoeren, ATC'25\n")
    print("=== CONFIGURATION ===")
    print(f"  Table entries:     {config.total_entries:,} "
          f"(Section 5.3: 100M, scaled 1/{SCALE_FACTOR} - ESTIMATION)")
    print(f"  Entries per row:   {config.entries_per_row} (Section 3.1: 8)")
    print(f"  Number of rows:    {config.num_rows:,}")
    print(f"  Entry size:        {config.entry_size_bytes}B "
          f"(Section 5.3: 8B = 32b key + 32b value)")
    print(f"  Locality f:        {config.locality_f} (Section 3.3: 2.3)")
    print(f"  Rows per lock:     {config.rows_per_lock} (Section 3.4.1: 16)")
    print(f"  Total locks:       {config.num_locks:,}")
    print(f"  Locks per MCAS:    {config.locks_per_mcas} (Section 3.4.1: 64)")
    print(f"  Max search depth:  {config.max_search_depth} (Section 3.2.3: 5)")
    print(f"  Client cache:      {config.client_cache_size_bytes//1024}KB "
          f"(Section 3.2.3: 64KB)")
    print(f"  Pre-populate:      {config.prepopulate_fill*100:.0f}% "
          f"(Section 5.3: 90%)")
    print(f"  Zipf theta:        {config.zipf_theta} (Section 5.3: 0.99)")
    print(f"  NIC bandwidth:     {config.nic_bandwidth_gbps:.0f} Gbps "
          f"(Section 2.2: 100 Gbps)")
    print(f"  NIC max reads/us:  {config.nic_max_read_ops_per_us} "
          f"(Figure 1b: ~75)")
    print(f"  NIC max CAS/us:    {config.nic_max_cas_ops_per_us} "
          f"(Figure 1b: ~50)")
    print(f"  RDMA RTT:          {config.rdma_rtt_us} us (ESTIMATION)")
    print(f"  Sim duration:      {config.sim_duration_us/1000:.0f}ms "
          f"(ESTIMATION)")
    print(f"  Client counts:     {config.figure6_client_counts}")
    print()

    results = run_figure6(config)
    print_results(results)
    plot_results(results)


if __name__ == "__main__":
    main()
