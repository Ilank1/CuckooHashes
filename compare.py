"""
Compare RCuckoo vs OfflineState on identical YCSB workloads.

Generates 9 plots:
1. Zipf distribution histogram
2. Throughput comparison (MOPS vs clients) — RCuckoo vs OfflineState
3. OfflineState read source breakdown (local/peer/server)
4. RDMA calls per operation vs clients
5. Read latency CDF (staircase: local/peer/server)
6. Bloom filter false positive rate vs clients
7. Throughput vs Zipf skewness (caching advantage)
8. Sync interval sensitivity (throughput, FPR, peer hit rate)
9. Cache size sensitivity (throughput, local hit%, RDMA/op)
"""

import os
from dataclasses import replace

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from rcuckoo.config import RCuckooConfig
from rcuckoo.table import IndexTable
from rcuckoo.workload import generate_workload, prepopulate
from rcuckoo.engine import run_simulation as rcuckoo_run_simulation

from offline_state.config import OfflineStateConfig
from offline_state.engine import run_simulation as offline_run_simulation

from sim_config import (
    SCALE_FACTOR, TOTAL_ENTRIES, ZIPF_THETA, RDMA_RTT_US, SERVER_RTT_TICKS,
    SIM_DURATION_US, NUM_TRIALS, CLIENT_COUNTS, ENTRY_SIZE_BYTES,
    OPS_PER_CLIENT, MIN_OPS,
    ENTRIES_PER_ROW, LOCALITY_F, ROWS_PER_LOCK, LOCKS_PER_MCAS,
    MAX_SEARCH_DEPTH, PREPOPULATE_FILL,
    CLIENT_CACHE_SIZE_BYTES, SYNC_INTERVAL_TICKS,
)

RCUCKOO_COLOR = '#4363d8'
OFFLINE_COLOR = '#3cb44b'
REFERENCE_COLOR = '#f58231'


def plot_zipf_distribution(key_samples, config, outdir):
    keys = key_samples[:1_000_000]

    fig, ax = plt.subplots(figsize=(8, 4))
    unique, counts = np.unique(keys, return_counts=True)
    top_idx = np.argsort(-counts)[:100]
    ax.bar(range(len(top_idx)), counts[top_idx] / len(keys) * 100, width=1.0,
           color=RCUCKOO_COLOR, alpha=0.8)
    ax.set_xlabel('Key rank (sorted by frequency)')
    ax.set_ylabel('Frequency (%)')
    ax.set_title(f'Zipf Distribution (theta={config.zipf_theta}, '
                 f'N={config.total_entries:,})')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(outdir, 'plot1_zipf_distribution.png')
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Plot saved: {path}")


def run_rcuckoo_evaluation(rcuckoo_config, workloads, client_counts,
                            pregenerated):
    results = {}
    stats_all = {}
    for wl in workloads:
        print(f"\n  RCuckoo {wl.upper()}")
        table = IndexTable(rcuckoo_config.num_rows,
                           rcuckoo_config.entries_per_row)
        prepopulate(table, rcuckoo_config, rcuckoo_config.prepopulate_fill)
        results[wl] = {}
        stats_all[wl] = {}
        for nc in client_counts:
            mops, stats = rcuckoo_run_simulation(
                rcuckoo_config, wl, nc, shared_table=table,
                pregenerated_workload=pregenerated[wl])
            results[wl][nc] = mops
            stats_all[wl][nc] = stats
    return results, stats_all


def run_offline_evaluation(offline_config, workloads, client_counts,
                            pregenerated):
    results = {}
    stats_all = {}
    for wl in workloads:
        print(f"\n  OfflineState {wl.upper()}")
        results[wl] = {}
        stats_all[wl] = {}
        for nc in client_counts:
            mops, stats = offline_run_simulation(
                offline_config, wl, nc,
                pregenerated_workload=pregenerated[wl])
            results[wl][nc] = mops
            stats_all[wl][nc] = stats
    return results, stats_all


def plot_throughput_comparison(rcuckoo_results, offline_results,
                               client_counts, outdir):
    workloads = ["ycsb-c", "ycsb-b", "ycsb-a"]
    titles = {
        "ycsb-c": "YCSB-C (100% read)",
        "ycsb-b": "YCSB-B (95% read)",
        "ycsb-a": "YCSB-A (50% read)",
    }

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

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for idx, wl in enumerate(workloads):
        ax = axes[idx]

        rc_clients = sorted(rcuckoo_results[wl].keys())
        rc_mops = [rcuckoo_results[wl][c] for c in rc_clients]
        ax.plot(rc_clients, rc_mops, 'o-', color=RCUCKOO_COLOR, linewidth=2,
                markersize=6, label='RCuckoo (sim)')

        os_clients = sorted(offline_results[wl].keys())
        os_mops = [offline_results[wl][c] for c in os_clients]
        ax.plot(os_clients, os_mops, 's-', color=OFFLINE_COLOR, linewidth=2,
                markersize=6, label='OfflineState (sim)')

        rd = reference_data[wl]
        ax.plot(rd["clients"], rd["mops"], '^--', color=REFERENCE_COLOR,
                linewidth=1.5, markersize=5, alpha=0.7,
                label='RCuckoo (reference)')

        # Ideal linear scaling reference (based on single-client RCuckoo)
        base_mops = rcuckoo_results[wl][min(rcuckoo_results[wl].keys())]
        base_nc = min(rcuckoo_results[wl].keys())
        ideal = [base_mops / base_nc * c for c in client_counts]
        ax.plot(client_counts, ideal, ':', color='gray', linewidth=1,
                alpha=0.5, label='Ideal linear')

        ax.set_xlabel('Clients')
        ax.set_ylabel('MOPS')
        ax.set_title(titles[wl])
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, max(client_counts) + 20)
        ax.set_ylim(bottom=0)

    plt.tight_layout()
    path = os.path.join(outdir, 'plot2_throughput_comparison.png')
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Plot saved: {path}")


def plot_read_source_breakdown(offline_stats, client_counts, outdir):
    workloads = ["ycsb-c", "ycsb-b", "ycsb-a"]
    titles = {
        "ycsb-c": "YCSB-C (100% read)",
        "ycsb-b": "YCSB-B (95% read)",
        "ycsb-a": "YCSB-A (50% read)",
    }

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for idx, wl in enumerate(workloads):
        ax = axes[idx]
        local_pct = []
        peer_pct = []
        server_pct = []

        for nc in client_counts:
            s = offline_stats[wl][nc]
            total = s['local_hits'] + s['peer_hits'] + s['server_hits']
            if total > 0:
                local_pct.append(s['local_hits'] / total * 100)
                peer_pct.append(s['peer_hits'] / total * 100)
                server_pct.append(s['server_hits'] / total * 100)
            else:
                local_pct.append(0)
                peer_pct.append(0)
                server_pct.append(0)

        x = np.arange(len(client_counts))
        width = 0.6

        ax.bar(x, local_pct, width, label='Local cache',
               color='#3cb44b')
        ax.bar(x, peer_pct, width, bottom=local_pct,
               label='Peer (broadcast)', color='#4363d8')
        ax.bar(x, server_pct, width,
               bottom=[l + p for l, p in zip(local_pct, peer_pct)],
               label='Server', color='#e6194B')

        ax.set_xlabel('Clients')
        ax.set_ylabel('Read source (%)')
        ax.set_title(titles[wl])
        ax.set_xticks(x)
        ax.set_xticklabels([str(c) for c in client_counts])
        ax.legend(loc='upper right', fontsize=8)
        ax.set_ylim(0, 105)
        ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    path = os.path.join(outdir, 'plot3_read_source_breakdown.png')
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Plot saved: {path}")


def plot_bloom_fpr(offline_stats, client_counts, outdir,
                   offline_config=None):
    """Plot 6: Bloom filter false positive rate vs clients + theoretical line."""
    import math

    workloads = ["ycsb-c", "ycsb-b", "ycsb-a"]
    titles = {
        "ycsb-c": "YCSB-C (100% read)",
        "ycsb-b": "YCSB-B (95% read)",
        "ycsb-a": "YCSB-A (50% read)",
    }

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for idx, wl in enumerate(workloads):
        ax = axes[idx]

        fpr_vals = []
        fpr_theory = []
        for nc in client_counts:
            s = offline_stats[wl][nc]
            fp = s.get('false_positives', 0)
            bloom_yes = fp + s['peer_hits']
            fpr_vals.append(fp / bloom_yes * 100 if bloom_yes > 0 else 0)

            # Theoretical FPR: p = (1 - e^(-kn/m))^k
            if offline_config:
                m, k = offline_config.effective_bloom_params(nc)
                n = offline_config.max_cache_entries * nc
                p = (1 - math.exp(-k * n / m)) ** k
                fpr_theory.append(p * 100)

        ax.plot(client_counts, fpr_vals, 's-', color=OFFLINE_COLOR,
                linewidth=2, markersize=6, label='Empirical')
        if fpr_theory:
            ax.plot(client_counts, fpr_theory, 's--', color=OFFLINE_COLOR,
                    linewidth=1, markersize=4, alpha=0.5,
                    label='Theoretical')

        ax.set_xlabel('Clients')
        ax.set_ylabel('Bloom FPR (%)')
        ax.set_title(titles[wl])
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, max(client_counts) + 20)
        ax.set_ylim(bottom=0)

    plt.tight_layout()
    path = os.path.join(outdir, 'plot6_bloom_fpr.png')
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Plot saved: {path}")


def plot_rdma_per_op(rcuckoo_stats, offline_stats, client_counts, outdir):
    """Plot 4: RDMA calls per operation vs clients."""
    workloads = ["ycsb-c", "ycsb-b", "ycsb-a"]
    titles = {
        "ycsb-c": "YCSB-C (100% read)",
        "ycsb-b": "YCSB-B (95% read)",
        "ycsb-a": "YCSB-A (50% read)",
    }

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for idx, wl in enumerate(workloads):
        ax = axes[idx]

        rc_rpo = []
        os_rpo = []
        for nc in client_counts:
            rs = rcuckoo_stats[wl][nc]
            rc_rpo.append(rs['total_rdma'] / rs['total_ops']
                          if rs['total_ops'] > 0 else 0)
            os_ = offline_stats[wl][nc]
            os_rpo.append(os_['total_rdma'] / os_['total_ops']
                          if os_['total_ops'] > 0 else 0)

        ax.plot(client_counts, rc_rpo, 'o-', color=RCUCKOO_COLOR,
                linewidth=2, markersize=6, label='RCuckoo')
        ax.plot(client_counts, os_rpo, 's-', color=OFFLINE_COLOR,
                linewidth=2, markersize=6, label='OfflineState')

        ax.set_xlabel('Clients')
        ax.set_ylabel('RDMA calls / op')
        ax.set_title(titles[wl])
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, max(client_counts) + 20)
        ax.set_ylim(bottom=0)

    plt.tight_layout()
    path = os.path.join(outdir, 'plot4_rdma_per_op.png')
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Plot saved: {path}")


def plot_latency_cdf(offline_stats, client_counts, config, outdir):
    """Plot 5: Read latency CDF — staircase showing local/peer/server tiers."""
    workloads = ["ycsb-c", "ycsb-b", "ycsb-a"]
    titles = {
        "ycsb-c": "YCSB-C (100% read)",
        "ycsb-b": "YCSB-B (95% read)",
        "ycsb-a": "YCSB-A (50% read)",
    }

    # Use highest client count for most dramatic effect
    nc = max(client_counts)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for idx, wl in enumerate(workloads):
        ax = axes[idx]
        s = offline_stats[wl][nc]
        lats = np.array(s.get('read_latencies', []))

        if len(lats) == 0:
            ax.set_title(titles[wl])
            continue

        sorted_lats = np.sort(lats)
        cdf = np.arange(1, len(sorted_lats) + 1) / len(sorted_lats)

        ax.plot(sorted_lats, cdf, '-', color=OFFLINE_COLOR, linewidth=2,
                label='OfflineState')

        # RCuckoo: all reads = server_rtt_ticks
        ax.axvline(x=config.server_rtt_ticks, color=RCUCKOO_COLOR,
                   linestyle='--', linewidth=2, label='RCuckoo (all reads)')

        # Annotate the tiers
        n_total = len(lats)
        n_local = np.sum(lats == 0)
        n_peer = np.sum(lats == 1)
        pct_local = n_local / n_total * 100
        pct_peer = n_peer / n_total * 100
        ax.annotate(f'Local\n{pct_local:.0f}%', xy=(0, pct_local / 100),
                    fontsize=8, color='#3cb44b', ha='center',
                    xytext=(0.8, pct_local / 100 - 0.05))
        if pct_peer > 2:
            ax.annotate(f'Peer\n{pct_peer:.0f}%',
                        xy=(1, (pct_local + pct_peer) / 100),
                        fontsize=8, color='#4363d8', ha='center',
                        xytext=(1.8, (pct_local + pct_peer) / 100 - 0.05))

        ax.set_xlabel('Latency (ticks)')
        ax.set_ylabel('CDF')
        ax.set_title(f'{titles[wl]} ({nc} clients)')
        ax.legend(loc='lower right', fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(-0.5, config.server_rtt_ticks + 1.5)
        ax.set_ylim(0, 1.05)

    plt.tight_layout()
    path = os.path.join(outdir, 'plot5_latency_cdf.png')
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Plot saved: {path}")


def plot_skewness_sensitivity(rcuckoo_config_base, offline_config_base,
                               outdir):
    """Plot 7: Throughput vs Zipf skewness — shows caching advantage."""
    thetas = [0.5, 0.7, 0.8, 0.9, 0.95, 0.99]
    nc = 80
    wl = "ycsb-c"

    rc_mops = []
    os_mops = []
    os_rdma_per_op = []

    for theta in thetas:
        print(f"    theta={theta:.2f} ...")
        num_ops = nc * OPS_PER_CLIENT
        key_samples, is_read = generate_workload(
            wl, offline_config_base.total_entries, theta, num_ops)
        pre = (key_samples, is_read)

        rc_cfg = replace(rcuckoo_config_base, zipf_theta=theta, num_trials=1)
        table = IndexTable(rc_cfg.num_rows, rc_cfg.entries_per_row)
        prepopulate(table, rc_cfg, rc_cfg.prepopulate_fill)
        mops_rc, _ = rcuckoo_run_simulation(
            rc_cfg, wl, nc, shared_table=table, pregenerated_workload=pre)
        rc_mops.append(mops_rc)

        os_cfg = replace(offline_config_base, zipf_theta=theta, num_trials=1)
        mops_os, stats_os = offline_run_simulation(
            os_cfg, wl, nc, pregenerated_workload=pre)
        os_mops.append(mops_os)
        os_rdma_per_op.append(
            stats_os['total_rdma'] / stats_os['total_ops']
            if stats_os['total_ops'] > 0 else 0)

        print(f"      RC={mops_rc:.1f} MOPS | OS={mops_os:.1f} MOPS | "
              f"OS RDMA/op={os_rdma_per_op[-1]:.3f}")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    axes[0].plot(thetas, rc_mops, 'o-', color=RCUCKOO_COLOR,
                 linewidth=2, markersize=6, label='RCuckoo')
    axes[0].plot(thetas, os_mops, 's-', color=OFFLINE_COLOR,
                 linewidth=2, markersize=6, label='OfflineState')
    axes[0].set_xlabel('Zipf skewness (θ)')
    axes[0].set_ylabel('MOPS')
    axes[0].set_title('Throughput')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[0].set_ylim(bottom=0)

    axes[1].plot(thetas, os_rdma_per_op, 's-', color=OFFLINE_COLOR,
                 linewidth=2, markersize=6, label='OfflineState')
    axes[1].axhline(y=1.0, color=RCUCKOO_COLOR, linestyle='--', linewidth=2,
                    label='RCuckoo (1.0)')
    axes[1].set_xlabel('Zipf skewness (θ)')
    axes[1].set_ylabel('RDMA calls / op')
    axes[1].set_title('RDMA per Operation')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    axes[1].set_ylim(bottom=0)

    fig.suptitle(f'Zipf Skewness Sensitivity (YCSB-C, {nc} clients)',
                 fontsize=13, y=1.02)
    plt.tight_layout()
    path = os.path.join(outdir, 'plot7_skewness_sensitivity.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Plot saved: {path}")


def plot_sync_sensitivity(config_base, outdir, pregenerated):
    """Plot 8: Sync interval sensitivity — throughput, FPR, peer hit rate."""
    sync_intervals = [10, 50, 100, 500, 1000, 5000]
    nc = 80
    wl = "ycsb-c"

    throughputs = []
    fprs = []
    peer_pcts = []

    for si in sync_intervals:
        cfg = replace(config_base, sync_interval_ticks=si, num_trials=1)
        mops, stats = offline_run_simulation(
            cfg, wl, nc, pregenerated_workload=pregenerated[wl])
        throughputs.append(mops)

        fp = stats.get('false_positives', 0)
        bloom_yes = fp + stats['peer_hits']
        fprs.append(fp / bloom_yes * 100 if bloom_yes > 0 else 0)

        total_reads = stats['local_hits'] + stats['peer_hits'] + stats['server_hits']
        peer_pcts.append(stats['peer_hits'] / total_reads * 100
                         if total_reads > 0 else 0)

        print(f"    sync={si:>5} ticks | {mops:.1f} MOPS | "
              f"FPR={fprs[-1]:.1f}% | peer={peer_pcts[-1]:.1f}%")

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    axes[0].plot(sync_intervals, throughputs, 'o-', color=OFFLINE_COLOR,
                 linewidth=2, markersize=6)
    axes[0].set_xlabel('Sync interval (ticks)')
    axes[0].set_ylabel('MOPS')
    axes[0].set_title('Throughput')
    axes[0].set_xscale('log')
    axes[0].grid(True, alpha=0.3)
    axes[0].set_ylim(bottom=0)

    axes[1].plot(sync_intervals, fprs, 'o-', color=OFFLINE_COLOR,
                 linewidth=2, markersize=6)
    axes[1].set_xlabel('Sync interval (ticks)')
    axes[1].set_ylabel('Bloom FPR (%)')
    axes[1].set_title('False Positive Rate')
    axes[1].set_xscale('log')
    axes[1].grid(True, alpha=0.3)
    axes[1].set_ylim(bottom=0)

    axes[2].plot(sync_intervals, peer_pcts, 'o-', color=RCUCKOO_COLOR,
                 linewidth=2, markersize=6)
    axes[2].set_xlabel('Sync interval (ticks)')
    axes[2].set_ylabel('Peer hit rate (%)')
    axes[2].set_title('Peer Cache Hit Rate')
    axes[2].set_xscale('log')
    axes[2].grid(True, alpha=0.3)
    axes[2].set_ylim(bottom=0)

    fig.suptitle(f'Sync Interval Sensitivity (YCSB-C, {nc} clients)',
                 fontsize=13, y=1.02)
    plt.tight_layout()
    path = os.path.join(outdir, 'plot8_sync_sensitivity.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Plot saved: {path}")


def plot_cache_sensitivity(config_base, outdir, pregenerated):
    """Plot 9: Cache size sensitivity — throughput, local hit%, RDMA/op."""
    cache_sizes_kb = [8, 16, 32, 64, 128, 256]
    nc = 80
    wl = "ycsb-c"

    throughputs = []
    local_pcts = []
    rdma_per_ops = []

    for cs_kb in cache_sizes_kb:
        cfg = replace(config_base, client_cache_size_bytes=cs_kb * 1024,
                      num_trials=1)
        mops, stats = offline_run_simulation(
            cfg, wl, nc, pregenerated_workload=pregenerated[wl])
        throughputs.append(mops)

        total_reads = stats['local_hits'] + stats['peer_hits'] + stats['server_hits']
        local_pcts.append(stats['local_hits'] / total_reads * 100
                          if total_reads > 0 else 0)
        rdma_per_ops.append(stats['total_rdma'] / stats['total_ops']
                            if stats['total_ops'] > 0 else 0)

        print(f"    cache={cs_kb:>4}KB | {mops:.1f} MOPS | "
              f"local={local_pcts[-1]:.1f}% | RDMA/op={rdma_per_ops[-1]:.3f}")

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    axes[0].plot(cache_sizes_kb, throughputs, 'o-', color=OFFLINE_COLOR,
                 linewidth=2, markersize=6)
    axes[0].set_xlabel('Cache size (KB)')
    axes[0].set_ylabel('MOPS')
    axes[0].set_title('Throughput')
    axes[0].grid(True, alpha=0.3)
    axes[0].set_ylim(bottom=0)

    axes[1].plot(cache_sizes_kb, local_pcts, 'o-', color='#3cb44b',
                 linewidth=2, markersize=6)
    axes[1].set_xlabel('Cache size (KB)')
    axes[1].set_ylabel('Local hit rate (%)')
    axes[1].set_title('Local Cache Hit Rate')
    axes[1].grid(True, alpha=0.3)
    axes[1].set_ylim(0, 105)

    axes[2].plot(cache_sizes_kb, rdma_per_ops, 'o-', color=OFFLINE_COLOR,
                 linewidth=2, markersize=6)
    axes[2].set_xlabel('Cache size (KB)')
    axes[2].set_ylabel('RDMA calls / op')
    axes[2].set_title('RDMA per Operation')
    axes[2].grid(True, alpha=0.3)
    axes[2].set_ylim(bottom=0)

    fig.suptitle(f'Cache Size Sensitivity (YCSB-C, {nc} clients)',
                 fontsize=13, y=1.02)
    plt.tight_layout()
    path = os.path.join(outdir, 'plot9_cache_sensitivity.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Plot saved: {path}")


def main():
    outdir = os.path.dirname(os.path.abspath(__file__))
    client_counts = CLIENT_COUNTS

    rcuckoo_config = RCuckooConfig(
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
        figure6_client_counts=client_counts,
    )

    offline_config = OfflineStateConfig(
        total_entries=TOTAL_ENTRIES,
        zipf_theta=ZIPF_THETA,
        client_cache_size_bytes=CLIENT_CACHE_SIZE_BYTES,
        entry_size_bytes=ENTRY_SIZE_BYTES,
        sync_interval_ticks=SYNC_INTERVAL_TICKS,
        rdma_rtt_us=RDMA_RTT_US,
        server_rtt_ticks=SERVER_RTT_TICKS,
        sim_duration_us=SIM_DURATION_US,
        num_trials=NUM_TRIALS,
        figure6_client_counts=client_counts,
    )

    print("=" * 60)
    print("COMPARISON: RCuckoo vs OfflineState")
    print("=" * 60)

    workloads = ["ycsb-c", "ycsb-b", "ycsb-a"]

    # Pre-generate workloads once so both systems use identical data
    max_ops = max(c * OPS_PER_CLIENT for c in client_counts)
    max_ops = max(max_ops, MIN_OPS)
    print(f"\n--- Generating workloads ({max_ops:,} ops each) ---")
    pregenerated = {}
    for wl in workloads:
        key_samples, is_read = generate_workload(
            wl, offline_config.total_entries, offline_config.zipf_theta, max_ops
        )
        pregenerated[wl] = (key_samples, is_read)
        print(f"  {wl.upper()}: {max_ops:,} keys generated")

    print("\n--- Plot 1: Zipf Distribution ---")
    plot_zipf_distribution(pregenerated["ycsb-c"][0], offline_config, outdir)

    print("\n--- Running RCuckoo Simulations ---")
    rcuckoo_results, rcuckoo_stats = run_rcuckoo_evaluation(
        rcuckoo_config, workloads, client_counts, pregenerated
    )

    print("\n--- Running OfflineState Simulations ---")
    offline_results, offline_stats = run_offline_evaluation(
        offline_config, workloads, client_counts, pregenerated
    )

    print("\n--- Plot 2: Throughput Comparison ---")
    plot_throughput_comparison(rcuckoo_results, offline_results,
                               client_counts, outdir)

    print("\n--- Plot 3: Read Source Breakdown ---")
    plot_read_source_breakdown(offline_stats, client_counts, outdir)

    print("\n--- Plot 4: RDMA per Operation ---")
    plot_rdma_per_op(rcuckoo_stats, offline_stats, client_counts, outdir)

    print("\n--- Plot 5: Read Latency CDF ---")
    plot_latency_cdf(offline_stats, client_counts, offline_config, outdir)

    print("\n--- Plot 6: Bloom FPR ---")
    plot_bloom_fpr(offline_stats, client_counts, outdir,
                   offline_config=offline_config)

    print("\n--- Plot 7: Skewness Sensitivity ---")
    plot_skewness_sensitivity(rcuckoo_config, offline_config, outdir)

    print("\n--- Plot 8: Sync Interval Sensitivity ---")
    plot_sync_sensitivity(offline_config, outdir, pregenerated)

    print("\n--- Plot 9: Cache Size Sensitivity ---")
    plot_cache_sensitivity(offline_config, outdir, pregenerated)

    # Summary table
    print(f"\n{'='*100}")
    print("THROUGHPUT (MOPS)")
    print(f"{'='*100}")
    header = f"{'Clients':>8}"
    for wl in workloads:
        header += f" | {'RC':>7} {'OS':>7}"
    print(header)
    print("-" * len(header))
    for nc in client_counts:
        row = f"{nc:>8}"
        for wl in workloads:
            rc = rcuckoo_results[wl].get(nc, 0)
            os_val = offline_results[wl].get(nc, 0)
            row += f" | {rc:>7.1f} {os_val:>7.1f}"
        print(row)
    print()

    print("RDMA/OP & FALSE POSITIVE RATE")
    print(f"{'='*100}")
    header2 = f"{'Clients':>8}"
    for wl in workloads:
        header2 += f" | {'RC':>7} {'OS':>7} {'FPR%':>6}"
    print(header2)
    print("-" * len(header2))
    for nc in client_counts:
        row = f"{nc:>8}"
        for wl in workloads:
            rs = rcuckoo_stats[wl].get(nc, {})
            os_ = offline_stats[wl].get(nc, {})
            rc_rpo = rs['total_rdma'] / rs['total_ops'] if rs.get('total_ops', 0) > 0 else 0
            os_rpo = os_['total_rdma'] / os_['total_ops'] if os_.get('total_ops', 0) > 0 else 0
            os_fp = os_.get('false_positives', 0)
            os_by = os_fp + os_.get('peer_hits', 0)
            os_fpr = os_fp / os_by * 100 if os_by > 0 else 0
            row += f" | {rc_rpo:>7.3f} {os_rpo:>7.3f} {os_fpr:>5.1f}%"
        print(row)
    print(f"{'='*100}")


if __name__ == "__main__":
    main()
