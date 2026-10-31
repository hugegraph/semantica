"""GraphStore integration layer for the native HugeGraph backend.

This module keeps the existing public ``GraphStore`` API intact. Only the
HugeGraph branch is added; all existing backends still execute the original
implementation unchanged.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Union

from ..utils.exceptions import ValidationError
from .graph_store import (
    GraphAnalytics as _BaseGraphAnalytics,
    GraphManager,
    GraphStore as _BaseGraphStore,
)
from .hugegraph_store import HugeGraphStore


class HugeGraphAnalytics(_BaseGraphAnalytics):
    """Expose HugeGraph native Traversers and Algorithm Jobs via analytics."""

    def degree_centrality(
        self,
        labels: Optional[List[str]] = None,
        rel_type: Optional[str] = None,
        direction: str = "both",
        **options: Any,
    ) -> Dict[str, Any]:
        parameters = dict(options.pop("parameters", {}) or {})
        if labels:
            raise ValidationError(
                "HugeGraph native degree_centrality does not support "
                "a vertex-label filter",
                field="labels",
                value=labels,
            )
        if rel_type:
            parameters.setdefault("label", rel_type)
        parameters.setdefault("direction", direction.upper())
        return self.backend.degree_centrality(**parameters, **options)

    def connected_components(
        self, labels: Optional[List[str]] = None, **options: Any
    ) -> Dict[str, Any]:
        parameters = dict(options.pop("parameters", {}) or {})
        if labels:
            raise ValidationError(
                "HugeGraph native weak_connected_component does not support "
                "a vertex-label filter",
                field="labels",
                value=labels,
            )
        return self.backend.connected_components(**parameters, **options)

    def page_rank(self, **options: Any) -> Dict[str, Any]:
        return self.backend.page_rank(**options)

    def personal_rank(
        self, source: Union[int, str], label: str, **options: Any
    ) -> Dict[str, float]:
        return self.backend.personal_rank(source, label, **options)

    def louvain(self, **options: Any) -> Dict[str, Any]:
        return self.backend.louvain(**options)

    def label_propagation(self, **options: Any) -> Dict[str, Any]:
        return self.backend.label_propagation(**options)

    def betweenness_centrality(self, **options: Any) -> Dict[str, Any]:
        return self.backend.betweenness_centrality(**options)

    def closeness_centrality(self, **options: Any) -> Dict[str, Any]:
        return self.backend.closeness_centrality(**options)

    def k_core(self, **options: Any) -> Dict[str, Any]:
        return self.backend.k_core(**options)

    def triangle_count(self, **options: Any) -> Dict[str, Any]:
        return self.backend.triangle_count(**options)

    def clustering_coefficient(self, **options: Any) -> Dict[str, Any]:
        return self.backend.clustering_coefficient(**options)

    def stress_centrality(self, **options: Any) -> Dict[str, Any]:
        return self.backend.stress_centrality(**options)

    def eigenvector_centrality(self, **options: Any) -> Dict[str, Any]:
        return self.backend.eigenvector_centrality(**options)

    def run_algorithm(
        self,
        name: str,
        parameters: Optional[Dict[str, Any]] = None,
        **options: Any,
    ) -> Dict[str, Any]:
        return self.backend.run_algorithm(name, parameters, **options)


class GraphStore(_BaseGraphStore):
    """Semantica GraphStore with a native ``hugegraph`` backend option."""

    HUGEGRAPH_ALIASES = {"hugegraph", "apache_hugegraph", "hg"}
    CONTEXT_EDGE_ID_PROPERTY = "_semantica_context_edge_id"
    CONTEXT_EDGE_FAMILY_PROPERTY = "_semantica_context_family_id"
    HUGEGRAPH_INTERNAL_ID_PROPERTY = "_semantica_hugegraph_internal_id"

    def _initialize_store_backend(self) -> None:
        if str(self.backend).lower() not in self.HUGEGRAPH_ALIASES:
            super()._initialize_store_backend()
            return

        config = self._hugegraph_config()
        config.update(self.config)
        config.pop("backend", None)
        self._store_backend = HugeGraphStore(**config)
        self._manager = GraphManager(self._store_backend)
        self._manager.analytics = HugeGraphAnalytics(self._store_backend)
        # Expose Reasoner write-back only on HugeGraph instances. Keeping
        # these as instance capabilities avoids changing getattr-based
        # behavior for every existing GraphStore backend.
        for method_name in (
            "add_fact",
            "assert_fact",
            "retract_fact",
            "remove_fact",
            "discard_fact",
        ):
            setattr(self, method_name, getattr(self._store_backend, method_name))

    @staticmethod
    def _hugegraph_config() -> Dict[str, Any]:
        graphspace = os.getenv("GRAPH_STORE_HUGEGRAPH_GRAPHSPACE")
        timeout_value = os.getenv("GRAPH_STORE_HUGEGRAPH_TIMEOUT")
        timeout: Any = None
        if timeout_value:
            try:
                timeout = float(timeout_value)
            except ValueError:
                timeout = timeout_value
        return {
            "url": os.getenv(
                "GRAPH_STORE_HUGEGRAPH_URL", "http://127.0.0.1:8080"
            ),
            "graph": os.getenv("GRAPH_STORE_HUGEGRAPH_GRAPH", "hugegraph"),
            "user": os.getenv("GRAPH_STORE_HUGEGRAPH_USER", "admin"),
            "password": os.getenv("GRAPH_STORE_HUGEGRAPH_PASSWORD", "admin"),
            "graphspace": graphspace,
            "timeout": timeout,
            "schema_mode": os.getenv(
                "GRAPH_STORE_HUGEGRAPH_SCHEMA_MODE", "auto"
            ),
            "candidate_limit": int(
                os.getenv("GRAPH_STORE_HUGEGRAPH_CANDIDATE_LIMIT", "400")
            ),
        }

    def _require_hugegraph(self) -> HugeGraphStore:
        if str(self.backend).lower() not in self.HUGEGRAPH_ALIASES:
            raise ValidationError(
                "This operation is specific to the HugeGraph backend",
                field="backend",
                value=self.backend,
            )
        return self._store_backend

    def query(
        self,
        query: str,
        parameters: Optional[Dict[str, Any]] = None,
        **options: Any,
    ) -> List[Dict[str, Any]]:
        if str(self.backend).lower() in self.HUGEGRAPH_ALIASES:
            return self._store_backend.query(query, parameters, **options)
        return super().query(query, parameters, **options)

    def get_node(
        self, node_id: Union[int, str], **options: Any
    ) -> Optional[Dict[str, Any]]:
        if str(self.backend).lower() in self.HUGEGRAPH_ALIASES:
            return self._store_backend.get_node(node_id, **options)
        return super().get_node(node_id, **options)

    def create_nodes(
        self, nodes: List[Dict[str, Any]], **options: Any
    ) -> List[Dict[str, Any]]:
        created = super().create_nodes(nodes, **options)
        if str(self.backend).lower() in self.HUGEGRAPH_ALIASES:
            # Keep application-id resolution consistent with create_node().
            for item in created:
                self._record_app_node_id(item)
        return created

    def update_relationship(
        self,
        rel_id: Union[int, str],
        properties: Dict[str, Any],
        merge: bool = True,
        **options: Any,
    ) -> Dict[str, Any]:
        """Update a HugeGraph edge without routing through Cypher."""
        if str(self.backend).lower() in self.HUGEGRAPH_ALIASES:
            return self._store_backend.update_relationship(
                rel_id, properties, merge=merge, **options
            )
        updater = getattr(super(), "update_relationship", None)
        if callable(updater):
            return updater(rel_id, properties, merge=merge, **options)
        raise ValidationError(
            "The selected graph backend does not expose relationship updates",
            field="backend",
            value=self.backend,
        )

    def create_relationships(
        self, relationships: List[Dict[str, Any]], **options: Any
    ) -> List[Dict[str, Any]]:
        """Create a relationship batch without changing other backends."""
        if str(self.backend).lower() in self.HUGEGRAPH_ALIASES:
            return self._store_backend.create_relationships(relationships, **options)
        return [
            self.create_relationship(
                relationship.get(
                    "start_node_id",
                    relationship.get("start_id", relationship.get("source")),
                ),
                relationship.get(
                    "end_node_id",
                    relationship.get("end_id", relationship.get("target")),
                ),
                relationship.get("type", relationship.get("rel_type", "RELATED_TO")),
                relationship.get("properties"),
                **options,
            )
            for relationship in relationships
        ]

    def add_edges(self, edges: List[Dict[str, Any]], **options: Any) -> int:
        """Use HugeGraph's transactional edge Batch API for context imports."""
        if str(self.backend).lower() not in self.HUGEGRAPH_ALIASES:
            return super().add_edges(edges, **options)

        relationships: List[Dict[str, Any]] = []
        for edge in edges:
            source_id = edge.get("source_id", edge.get("source"))
            target_id = edge.get("target_id", edge.get("target"))
            if source_id is None or target_id is None:
                continue
            properties = dict(edge.get("properties") or {})
            if "weight" in edge:
                properties.setdefault("weight", edge["weight"])
            relationships.append(
                {
                    "start_node_id": source_id,
                    "end_node_id": target_id,
                    "type": edge.get("type", "RELATED_TO"),
                    "properties": properties,
                }
            )
        return len(self.create_relationships(relationships, **options))

    @staticmethod
    def _context_item_to_dict(item: Any) -> Dict[str, Any]:
        if isinstance(item, dict):
            return dict(item)
        converter = getattr(item, "to_dict", None)
        if callable(converter):
            converted = converter()
            if isinstance(converted, dict):
                return dict(converted)
        values = getattr(item, "__dict__", None)
        return dict(values) if isinstance(values, dict) else {}

    def import_context_graph(
        self, context_graph: Any, **options: Any
    ) -> Dict[str, Any]:
        """Persist an existing Semantica ``ContextGraph`` into HugeGraph.

        This is an explicit import operation; normal retrieval reads the
        external HugeGraph graph directly and does not require a full copy.
        """
        if isinstance(context_graph, dict):
            raw_nodes = context_graph.get("nodes", context_graph.get("entities", []))
            raw_edges = context_graph.get(
                "edges", context_graph.get("relationships", [])
            )
        else:
            raw_nodes = getattr(context_graph, "nodes", [])
            raw_edges = getattr(context_graph, "edges", [])

        if isinstance(raw_nodes, dict):
            raw_nodes = raw_nodes.values()
        nodes: List[Dict[str, Any]] = []
        for raw in raw_nodes or []:
            item = self._context_item_to_dict(raw)
            node_id = item.get("id", item.get("node_id"))
            if node_id is None:
                continue
            properties = dict(item.get("properties") or item.get("metadata") or {})
            content = item.get("content")
            if content is not None:
                properties.setdefault("content", content)
            for temporal_key in ("valid_from", "valid_until"):
                if item.get(temporal_key) is not None:
                    properties.setdefault(temporal_key, item[temporal_key])
            nodes.append(
                {
                    "id": node_id,
                    "type": item.get("type", item.get("node_type", "Entity")),
                    "properties": properties,
                }
            )

        edges: List[Dict[str, Any]] = []
        for raw in raw_edges or []:
            item = self._context_item_to_dict(raw)
            source = item.get("source_id", item.get("source"))
            target = item.get("target_id", item.get("target"))
            if source is None or target is None:
                continue
            properties = dict(item.get("properties") or item.get("metadata") or {})
            if "weight" in item:
                properties.setdefault("weight", item["weight"])
            original_edge_id = item.get("id", item.get("edge_id"))
            family_id = item.get("familyId", item.get("family_id"))
            if original_edge_id is not None:
                properties.setdefault(
                    self.CONTEXT_EDGE_ID_PROPERTY, str(original_edge_id)
                )
            if family_id is not None:
                properties.setdefault(
                    self.CONTEXT_EDGE_FAMILY_PROPERTY, str(family_id)
                )
            for temporal_key in ("valid_from", "valid_until"):
                if item.get(temporal_key) is not None:
                    properties.setdefault(temporal_key, item[temporal_key])
            edges.append(
                {
                    "source_id": source,
                    "target_id": target,
                    "type": item.get("type", item.get("edge_type", "RELATED_TO")),
                    "properties": properties,
                }
            )

        node_count = self.add_nodes(nodes, **options)
        edge_count = self.add_edges(edges, **options)
        return {"statistics": {"node_count": node_count, "edge_count": edge_count}}

    persist_context_graph = import_context_graph

    def to_context_graph(
        self,
        seed_ids: List[Union[int, str]],
        depth: int = 1,
        *,
        advanced_analytics: bool = False,
        **options: Any,
    ) -> Any:
        """Materialize only a bounded HugeGraph subgraph as ``ContextGraph``."""
        from ..context import ContextGraph

        self._require_hugegraph()
        subgraph = self.load_subgraph(seed_ids, depth=depth, **options)
        graph = ContextGraph(advanced_analytics=advanced_analytics)

        # External HugeGraph labels may use AUTOMATIC/PRIMARY_KEY internal IDs
        # while exposing the stable business identifier in the conventional
        # ``id`` property. Preserve that business identifier in ContextGraph
        # whenever it is unique inside the bounded materialized subgraph.
        candidate_ids = []
        candidate_counts: Dict[Any, int] = {}
        for node in subgraph["nodes"]:
            candidate_id = (node.get("properties") or {}).get("id", node["id"])
            if not isinstance(candidate_id, (str, int)) or isinstance(
                candidate_id, bool
            ):
                candidate_id = node["id"]
            candidate_ids.append(candidate_id)
            candidate_counts[candidate_id] = candidate_counts.get(candidate_id, 0) + 1
        duplicate_ids = {
            value for value, count in candidate_counts.items() if count > 1
        }
        node_id_map: Dict[Any, Any] = {}
        context_nodes: List[Dict[str, Any]] = []
        for node, candidate_id in zip(subgraph["nodes"], candidate_ids):
            context_id = node["id"] if candidate_id in duplicate_ids else candidate_id
            node_id_map[node["id"]] = context_id
            properties = dict(node.get("properties") or {})
            if context_id != node["id"]:
                properties.setdefault(self.HUGEGRAPH_INTERNAL_ID_PROPERTY, node["id"])
            context_nodes.append(
                {
                    "id": context_id,
                    "type": node.get("type", "Entity"),
                    "properties": properties,
                    "content": properties.get("content", ""),
                }
            )
        graph.add_nodes(context_nodes)
        context_edges: List[Dict[str, Any]] = []
        for edge in subgraph["relationships"]:
            properties = dict(edge.get("properties") or {})
            context_edge_id = properties.pop(
                self.CONTEXT_EDGE_ID_PROPERTY, edge.get("id")
            )
            family_id = properties.pop(
                self.CONTEXT_EDGE_FAMILY_PROPERTY, context_edge_id
            )
            context_edges.append(
                {
                    "id": context_edge_id,
                    "familyId": family_id,
                    "source_id": node_id_map.get(
                        edge["start_node_id"], edge["start_node_id"]
                    ),
                    "target_id": node_id_map.get(
                        edge["end_node_id"], edge["end_node_id"]
                    ),
                    "type": edge.get("type", "RELATED_TO"),
                    "weight": properties.get("weight", 1.0),
                    "valid_from": properties.get("valid_from"),
                    "valid_until": properties.get("valid_until"),
                    "properties": properties,
                }
            )
        graph.add_edges(context_edges)
        return graph

    load_context_graph = to_context_graph

    def run_algorithm(
        self,
        name: str,
        parameters: Optional[Dict[str, Any]] = None,
        **options: Any,
    ) -> Dict[str, Any]:
        return self._require_hugegraph().run_algorithm(name, parameters, **options)

    def get_algorithm_task(self, task_id: Union[int, str]) -> Dict[str, Any]:
        return self._require_hugegraph().get_algorithm_task(task_id)

    def personal_rank(
        self, source: Union[int, str], label: str, **options: Any
    ) -> Dict[str, float]:
        return self._require_hugegraph().personal_rank(source, label, **options)

    def k_out(self, source: Union[int, str], depth: int, **options: Any) -> Any:
        return self._require_hugegraph().k_out(source, depth, **options)

    def load_subgraph(
        self,
        seed_ids: List[Union[int, str]],
        depth: int = 1,
        **options: Any,
    ) -> Dict[str, List[Dict[str, Any]]]:
        return self._require_hugegraph().load_subgraph(seed_ids, depth, **options)
