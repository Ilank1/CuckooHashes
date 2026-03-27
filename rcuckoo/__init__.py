"""
RCuckoo Simulator

Tick-based simulation of the RCuckoo disaggregated cuckoo hash table.
Each tick represents one RDMA round trip. All clients advance in parallel.
"""

from rcuckoo.config import RCuckooConfig
from rcuckoo.engine import run_simulation
from rcuckoo.evaluation import run_figure6, print_results, plot_results
