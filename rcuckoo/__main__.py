"""
Entry point for running the RCuckoo simulator.

Usage:
    python -m rcuckoo
"""

from rcuckoo.config import RCuckooConfig
from rcuckoo.evaluation import run_figure6, print_results, plot_results
from sim_config import (
    SCALE_FACTOR, TOTAL_ENTRIES, ENTRIES_PER_ROW, LOCALITY_F, ROWS_PER_LOCK,
    LOCKS_PER_MCAS, MAX_SEARCH_DEPTH, PREPOPULATE_FILL,
    ZIPF_THETA, RDMA_RTT_US, SERVER_RTT_TICKS,
    SIM_DURATION_US, NUM_TRIALS, CLIENT_COUNTS,
)


def main():
    config = RCuckooConfig(
        total_entries=TOTAL_ENTRIES,
        entries_per_row=ENTRIES_PER_ROW,
        locality_f=LOCALITY_F,
        rows_per_lock=ROWS_PER_LOCK,
        locks_per_mcas=LOCKS_PER_MCAS,
        max_search_depth=MAX_SEARCH_DEPTH,
        prepopulate_fill=PREPOPULATE_FILL,
        zipf_theta=ZIPF_THETA,
        rdma_rtt_us=RDMA_RTT_US,
        server_rtt_ticks=SERVER_RTT_TICKS,
        sim_duration_us=SIM_DURATION_US,
        num_trials=NUM_TRIALS,
        figure6_client_counts=CLIENT_COUNTS,
    )

    print("RCuckoo Simulator")
    print(f"  Table entries:     {config.total_entries:,} (scaled 1/{SCALE_FACTOR})")
    print(f"  Entries per row:   {config.entries_per_row}")
    print(f"  Number of rows:    {config.num_rows:,}")
    print(f"  Locality f:        {config.locality_f}")
    print(f"  Rows per lock:     {config.rows_per_lock}")
    print(f"  Total locks:       {config.num_locks:,}")
    print(f"  Locks per MCAS:    {config.locks_per_mcas}")
    print(f"  Max search depth:  {config.max_search_depth}")
    print(f"  Client cache:      {config.client_cache_size_bytes//1024}KB")
    print(f"  Pre-populate:      {config.prepopulate_fill*100:.0f}%")
    print(f"  Zipf theta:        {config.zipf_theta}")
    print(f"  RDMA RTT:          {config.rdma_rtt_us} us")
    print(f"  Sim duration:      {config.sim_duration_us/1000:.0f}ms")
    print(f"  Client counts:     {config.figure6_client_counts}")
    print()

    results = run_figure6(config)
    print_results(results)
    plot_results(results)


if __name__ == "__main__":
    main()
