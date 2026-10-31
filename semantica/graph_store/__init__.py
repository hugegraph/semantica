"""Unified property-graph backends for Semantica.

The public :class:`GraphStore` keeps the existing Neo4j, FalkorDB, Amazon
Neptune and Apache AGE behavior and additionally exposes a native Apache
HugeGraph backend. HugeGraph uses its REST, Gremlin, Traverser and Algorithm
Job APIs directly; no Cypher translation layer is involved.
"""

from . import graph_store as _graph_store_module
from .age_store import ApacheAgeStore
from .amazon_neptune import (
    AmazonNeptuneStore,
    NeptuneAuthTokenManager,
    NeptuneDriver,
    NeptuneSession,
    NeptuneTransaction,
)
from .config import GraphStoreConfig, graph_store_config
from .falkordb_store import FalkorDBClient, FalkorDBGraph, FalkorDBStore
from .graph_store import (
    GraphAnalytics,
    GraphManager,
    NodeManager,
    QueryEngine,
    RelationshipManager,
)
from .hugegraph_graph_store import GraphStore, HugeGraphAnalytics
from .hugegraph_store import HugeGraphStore

# ``methods.py`` and callers that import the implementation module directly
# still resolve the enhanced class. This avoids duplicating the mature common
# GraphStore implementation while keeping every existing backend unchanged.
_graph_store_module.GraphStore = GraphStore

from .methods import (  # noqa: E402 - import after compatibility binding
    create_node,
    create_nodes,
    create_relationship,
    create_relationships,
    delete_node,
    delete_relationship,
    execute_query,
    get_graph_store_method,
    get_neighbors,
    get_nodes,
    get_relationships,
    list_available_methods,
    run_analytics,
    shortest_path,
    update_node,
    update_relationship,
)
from .neo4j_store import Neo4jDriver, Neo4jStore, Neo4jTransaction
from .registry import MethodRegistry, method_registry

__all__ = [
    # Core graph store
    "GraphStore",
    "GraphManager",
    "NodeManager",
    "RelationshipManager",
    "QueryEngine",
    "GraphAnalytics",
    "HugeGraphAnalytics",
    # HugeGraph
    "HugeGraphStore",
    # Neo4j
    "Neo4jStore",
    "Neo4jDriver",
    "Neo4jTransaction",
    # Apache AGE
    "ApacheAgeStore",
    # Amazon Neptune
    "AmazonNeptuneStore",
    "NeptuneAuthTokenManager",
    "NeptuneDriver",
    "NeptuneSession",
    "NeptuneTransaction",
    # FalkorDB
    "FalkorDBStore",
    "FalkorDBClient",
    "FalkorDBGraph",
    # Convenience functions
    "create_node",
    "create_nodes",
    "create_relationship",
    "create_relationships",
    "get_nodes",
    "get_relationships",
    "get_neighbors",
    "update_node",
    "update_relationship",
    "delete_node",
    "delete_relationship",
    "execute_query",
    "shortest_path",
    "run_analytics",
    "get_graph_store_method",
    "list_available_methods",
    # Configuration and registry
    "GraphStoreConfig",
    "graph_store_config",
    "MethodRegistry",
    "method_registry",
]
