"""
RCuckoo Simulator

Event-driven simulation of the RCuckoo disaggregated cuckoo hash table.
Based on "Cuckoo for Clients: Disaggregated Cuckoo Hashing"
by Stewart Grant and Alex C. Snoeren (USENIX ATC'25).

Each tick represents one RDMA round trip. All clients advance in parallel
per tick, modeling RDMA parallelism.
"""

from rcuckoo.config import RCuckooConfig
from rcuckoo.engine import run_simulation
from rcuckoo.evaluation import run_figure6, print_results, plot_results
