"""
OfflineState Distributed Cache Simulator

Client-side LRU caches with bloom filters and periodic offline sync.
Reads check: local cache (0 RTT) -> peer broadcast (1 RTT) -> server (1 RTT).
Updates use one-sided RDMA write to server (1 RTT).
"""

from offline_state.config import OfflineStateConfig
from offline_state.engine import run_simulation
