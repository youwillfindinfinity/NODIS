"""
nodis.network — Network topology analysis for inferred GGMs.

Provides community detection, hub gene identification, and backbone
extraction on precision matrices and adjacency matrices produced by
NODIS estimators.

Public API
----------
NetworkTopology       — main class: communities, hubs, backbone
CommunityResult       — structured result for community detection
HubResult             — structured result for hub identification
to_networkx           — convert adjacency/precision to networkx Graph
write_anndata_network — write topology results into AnnData slots
"""

from nodis.network.topology import (
    NetworkTopology,
    CommunityResult,
    HubResult,
    to_networkx,
    write_anndata_network,
)

__all__ = [
    "NetworkTopology",
    "CommunityResult",
    "HubResult",
    "to_networkx",
    "write_anndata_network",
]
