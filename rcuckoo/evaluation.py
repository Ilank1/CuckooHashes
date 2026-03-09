"""
Figure 6 evaluation (Section 5.3).

Grant & Snoeren, ATC'25:
    "Figure 6 shows YCSB throughput for RCuckoo, FUSEE, Clover, and
    Sherman on three different YCSB workloads."

Runs YCSB-A/B/C across varying client counts and plots simulation
results against reported reference values.
"""

import numpy as np

from rcuckoo.config import RCuckooConfig
from rcuckoo.table import IndexTable
from rcuckoo.workload import prepopulate
from rcuckoo.engine import run_simulation


def run_figure6(config: RCuckooConfig):
    """
    Reproduce Figure 6: throughput (MOPS) vs number of clients
    for YCSB-C, YCSB-B, and YCSB-A.
    """
    workloads = ["ycsb-c", "ycsb-b", "ycsb-a"]
    client_counts = config.figure6_client_counts
    results = {wl: {} for wl in workloads}

    for wl in workloads:
        print(f"\n{'='*60}")
        print(f"WORKLOAD: {wl.upper()}")
        print(f"{'='*60}")

        # Pre-create table once per workload (updates are negligible
        # relative to table size)
        table = IndexTable(config.num_rows, config.entries_per_row)
        prepopulate(table, config, config.prepopulate_fill)

        for nc in client_counts:
            mops_trials = []
            for trial in range(config.num_trials):
                mops = run_simulation(config, wl, nc, shared_table=table)
                mops_trials.append(mops)
            results[wl][nc] = np.mean(mops_trials)

    return results


def print_results(results: dict):
    """Print a formatted results table."""
    print(f"\n{'='*70}")
    print("FIGURE 6 RESULTS: Throughput (MOPS) vs Number of Clients")
    print(f"{'='*70}")

    workloads = list(results.keys())
    header = f"{'Clients':>8}"
    for wl in workloads:
        header += f" | {wl.upper():>12}"
    print(header)
    print("-" * len(header))

    all_clients = sorted(set(
        c for wl_results in results.values() for c in wl_results.keys()
    ))
    for nc in all_clients:
        row = f"{nc:>8}"
        for wl in workloads:
            mops = results[wl].get(nc, 0)
            row += f" | {mops:>12.3f}"
        print(row)
    print(f"{'='*70}")


def plot_results(results: dict):
    """Plot Figure 6 style charts comparing simulation to reference values."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        titles = {
            "ycsb-c": "(a) Read only (YCSB-C)",
            "ycsb-b": "(b) 95% read, 5% update (YCSB-B)",
            "ycsb-a": "(c) 50% read, 50% update (YCSB-A)"
        }
        # Reference throughput from Section 5.3, Figure 6
        reference_data = {
            "ycsb-c": {
                "clients": [10, 20, 40, 80, 160, 320],
                "mops": [2.99, 5.953, 11.494, 22.579, 39.369, 46.493]
            },
            "ycsb-b": {
                "clients": [8, 16, 40, 80, 160, 320],
                "mops": [1.936, 3.764, 8.914, 16.538, 27.736, 38.570]
            },
            "ycsb-a": {
                "clients": [8, 16, 40, 80, 160, 320],
                "mops": [0.909, 1.819, 4.275, 7.860, 13.809, 22.353]
            },
        }

        for idx, wl in enumerate(["ycsb-c", "ycsb-b", "ycsb-a"]):
            ax = axes[idx]
            wl_results = results[wl]
            clients = sorted(wl_results.keys())
            mops = [wl_results[c] for c in clients]

            ax.plot(clients, mops, 'o-', color='#4363d8', linewidth=2,
                    markersize=6, label='RCuckoo (sim)')

            rd = reference_data[wl]
            ax.plot(rd["clients"], rd["mops"], 's--', color='#e6194B',
                    linewidth=1.5, markersize=5, alpha=0.7,
                    label='RCuckoo (reference)')

            ax.set_xlabel('clients')
            ax.set_ylabel('MOPS')
            ax.set_title(titles[wl])
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.set_xlim(0, max(clients) + 20)
            ax.set_ylim(bottom=0)

        plt.tight_layout()
        import os
        out = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                           'figure6_results.png')
        plt.savefig(out, dpi=150)
        print(f"\nPlot saved to {out}")
    except Exception as e:
        print(f"\nCould not plot: {e}")
