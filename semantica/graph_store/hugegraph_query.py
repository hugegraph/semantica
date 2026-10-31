"""Gremlin, GraphRAG context retrieval, reasoning hooks and graph traversal."""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

from ..utils.exceptions import ProcessingError, ValidationError


NodeId = Union[int, str]


class HugeGraphQueryMixin:
    """Internal implementation mixin; composed by :class:`HugeGraphStore`."""

    @classmethod
    def _gremlin_literal(cls, value: Any) -> str:
        if value is None:
            return "null"
        if isinstance(value, bool):
            return "true" if value else "false"
        if isinstance(value, (int, float)):
            return repr(value)
        if isinstance(value, str):
            return json.dumps(value, ensure_ascii=False)
        if isinstance(value, (list, tuple, set)):
            return "[" + ",".join(cls._gremlin_literal(v) for v in value) + "]"
        if isinstance(value, Mapping):
            return cls._gremlin_literal(cls._canonical_json(value))
        return cls._gremlin_literal(str(value))

    @classmethod
    def _bind_gremlin(cls, query: str, parameters: Mapping[str, Any]) -> str:
        rendered = query
        for name in sorted(parameters, key=len, reverse=True):
            if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name):
                raise ValidationError(
                    "Invalid Gremlin parameter name", field="parameter", value=name
                )
            rendered = re.sub(
                rf"\${re.escape(name)}\b",
                lambda _: cls._gremlin_literal(parameters[name]),
                rendered,
            )
        remaining = re.findall(r"\$[A-Za-z_][A-Za-z0-9_]*", rendered)
        if remaining:
            raise ValidationError(
                "Missing Gremlin parameters", missing=sorted(set(remaining))
            )
        return rendered

    def execute_query(
        self,
        query: str,
        parameters: Optional[Dict[str, Any]] = None,
        **options: Any,
    ) -> Dict[str, Any]:
        """Execute native Gremlin and return Semantica's query envelope."""
        language = str(options.pop("language", "gremlin")).lower()
        if language not in {"gremlin", "groovy"}:
            raise ValidationError(
                "HugeGraphStore only accepts native Gremlin queries",
                field="language",
                value=language,
            )
        stripped = query.lstrip()
        if not stripped.startswith(("g.", "__.", "graph.")):
            raise ValidationError(
                "HugeGraphStore does not translate Cypher; provide native Gremlin",
                field="query",
                value=query[:80],
            )
        rendered = self._bind_gremlin(query, parameters or {})
        try:
            raw = self._manager("gremlin").exec(rendered)
        except Exception as exc:
            raise ProcessingError(
                "HugeGraph Gremlin query failed",
                processing_context={"query": rendered[:500]},
            ) from exc
        data = self._gremlin_data(raw)
        records: List[Dict[str, Any]] = []
        for value in data:
            if isinstance(value, Mapping):
                records.append(dict(value))
            else:
                records.append({"result": value})
        keys: List[str] = []
        for record in records:
            for key in record:
                if key not in keys:
                    keys.append(key)
        return {
            "success": True,
            "records": records,
            "keys": keys,
            "metadata": {
                "backend": "hugegraph",
                "language": "gremlin",
                "raw_meta": raw.get("result", {}).get("meta")
                if isinstance(raw, Mapping)
                and isinstance(raw.get("result"), Mapping)
                else None,
            },
        }

    def query_context(
        self,
        query: str,
        limit: int = 20,
        labels: Optional[Sequence[str]] = None,
        candidate_limit: Optional[int] = None,
        **options: Any,
    ) -> List[Dict[str, Any]]:
        """Return bounded, server-fetched candidates for ContextRetriever.

        HugeGraph performs the bounded candidate read; only that local
        subgraph is scored in Python. This keeps direct external-graph access
        possible without copying the entire graph into memory. Existing
        text/vector stores can still rerank and fuse these results in
        ``ContextRetriever``.
        """
        limit = max(1, int(limit))
        bounded = max(limit, int(candidate_limit or self.candidate_limit))
        traversal = "g.V()"
        if labels:
            safe_labels = [
                self._gremlin_literal(self._schema_name(label, prefix="vertex"))
                for label in labels
            ]
            traversal += f".hasLabel({','.join(safe_labels)})"
        traversal += f".limit({bounded}).elementMap()"
        raw = self._manager("gremlin").exec(traversal)
        candidates = [
            self._normalize_element_map(value) for value in self._gremlin_data(raw)
        ]
        tokens = {
            token.lower()
            for token in self._TOKEN_RE.findall(query)
            if len(token) > 1 and token.lower() not in self._STOP_WORDS
        }
        results: List[Dict[str, Any]] = []
        for node in candidates:
            if not node:
                continue
            props = node["properties"]
            content_parts: List[str] = []
            for key in self.content_properties:
                value = props.get(self._property_name(key), props.get(key))
                if value is not None:
                    content_parts.append(
                        value if isinstance(value, str) else self._canonical_json(value)
                    )
            if not content_parts:
                for key, value in props.items():
                    if key.startswith(self.RESERVED_PREFIX):
                        continue
                    if isinstance(value, (str, int, float, bool)):
                        content_parts.append(str(value))
            content = "\n".join(part for part in content_parts if part)
            haystack = f"{node['type']} {content} {self._canonical_json(props)}".lower()
            if tokens:
                matched = sum(1 for token in tokens if token in haystack)
                score = matched / len(tokens)
            else:
                score = 0.5
            if tokens and score == 0:
                continue
            metadata = dict(props)
            metadata.update(
                {
                    "node_id": node["id"],
                    "labels": node["labels"],
                    "backend": "hugegraph",
                }
            )
            results.append(
                {
                    "id": node["id"],
                    "type": node["type"],
                    "content": content,
                    "metadata": metadata,
                    "score": score,
                    "node": node,
                }
            )
        results.sort(key=lambda item: (item["score"], str(item["id"])), reverse=True)
        return results[:limit]

    def query(
        self, query: str, parameters: Optional[Dict[str, Any]] = None, **options: Any
    ) -> List[Dict[str, Any]]:
        """ContextRetriever-compatible natural-language graph search."""
        language = options.pop("language", None)
        normalized_language = str(language).lower() if language is not None else None
        if normalized_language in {"gremlin", "groovy"} or query.lstrip().startswith(
            ("g.", "__.", "graph.")
        ):
            return self.execute_query(
                query, parameters, language="gremlin", **options
            ).get("records", [])
        if normalized_language not in {None, "context", "natural", "text"}:
            raise ValidationError(
                "HugeGraphStore.query supports natural-language context search "
                "or native Gremlin only",
                field="language",
                value=language,
            )
        return self.query_context(query, **options)

    @classmethod
    def _parse_fact(cls, fact: str) -> Tuple[str, List[str]]:
        match = cls._FACT_RE.match(str(fact))
        if match is None:
            raise ValidationError(
                "HugeGraph reasoning facts must use Predicate(arg1, arg2) syntax",
                field="fact",
                value=fact,
            )
        predicate = match.group(1)
        inner = match.group(2).strip()
        arguments = [] if not inner else [part.strip() for part in inner.split(",")]
        arguments = [
            arg[1:-1]
            if len(arg) >= 2 and arg[0] == arg[-1] and arg[0] in {"'", '"'}
            else arg
            for arg in arguments
        ]
        if len(arguments) not in {1, 2} or any(not arg for arg in arguments):
            raise ValidationError(
                "HugeGraph reasoning write-back supports unary and binary facts",
                field="fact",
                value=fact,
            )
        return predicate, arguments

    def add_fact(self, fact: str) -> Dict[str, Any]:
        """Persist a Semantica ``Reasoner`` fact using native graph elements.

        Unary facts are attached to the referenced vertex; a missing vertex is
        created with the predicate as its logical label. Binary facts are
        represented as typed, directed edges. A reserved fact marker makes
        repeated assertions idempotent and gives ``retract_fact`` an exact
        deletion target without scanning or rewriting unrelated graph data.
        """
        predicate, arguments = self._parse_fact(fact)
        if len(arguments) == 1:
            node_id = arguments[0]
            node = self.get_node(node_id)
            if node is None:
                return self.create_node(
                    [predicate],
                    {
                        self.APP_ID_PROPERTY: node_id,
                        "content": node_id,
                        self.FACTS_PROPERTY: [fact],
                        self.FACT_OWNED_PROPERTY: True,
                    },
                )
            existing = node["properties"].get(self.FACTS_PROPERTY, [])
            facts = list(existing) if isinstance(existing, list) else [existing]
            if fact not in facts:
                facts.append(fact)
                node = self.update_node(node["id"], {self.FACTS_PROPERTY: facts})
            return node

        source_id, target_id = arguments
        source = self.get_node(source_id)
        if source is None:
            source = self.create_node(
                ["Entity"],
                {self.APP_ID_PROPERTY: source_id, "content": source_id},
            )
        target = self.get_node(target_id)
        if target is None:
            target = self.create_node(
                ["Entity"],
                {self.APP_ID_PROPERTY: target_id, "content": target_id},
            )
        for edge in self.get_relationships(
            source["id"], rel_type=predicate, direction="out", limit=10_000
        ):
            values = edge["properties"].get(self.FACTS_PROPERTY, [])
            facts = values if isinstance(values, list) else [values]
            if edge.get("end_node_id") == target["id"] and fact in facts:
                return edge
        return self.create_relationship(
            source["id"],
            target["id"],
            predicate,
            {self.FACTS_PROPERTY: [fact], "content": fact},
        )

    def retract_fact(self, fact: str) -> bool:
        """Remove only graph state created for the exact reasoning fact."""
        predicate, arguments = self._parse_fact(fact)
        if len(arguments) == 2:
            source = self.get_node(arguments[0])
            target = self.get_node(arguments[1])
            if source is None or target is None:
                return False
            removed = False
            for edge in self.get_relationships(
                source["id"], rel_type=predicate, direction="out", limit=10_000
            ):
                values = edge["properties"].get(self.FACTS_PROPERTY, [])
                facts = values if isinstance(values, list) else [values]
                if edge.get("end_node_id") == target["id"] and fact in facts:
                    self.delete_relationship(edge["id"])
                    removed = True
            return removed

        node = self.get_node(arguments[0])
        if node is None:
            return False
        values = node["properties"].get(self.FACTS_PROPERTY, [])
        facts = list(values) if isinstance(values, list) else [values]
        if fact not in facts:
            return False
        remaining = [item for item in facts if item != fact]
        owned = bool(node["properties"].get(self.FACT_OWNED_PROPERTY))
        if owned and not remaining and not self.get_relationships(
            node["id"], direction="both", limit=1
        ):
            return self.delete_node(node["id"])
        graph = self._manager("graph")
        if remaining:
            self.update_node(node["id"], {self.FACTS_PROPERTY: remaining})
        else:
            graph.eliminateVertex(node["id"], {self.FACTS_PROPERTY: [fact]})
        return True

    def get_neighbors(
        self,
        node_id: NodeId,
        rel_type: Optional[str] = None,
        direction: str = "both",
        depth: int = 1,
        **options: Any,
    ) -> List[Dict[str, Any]]:
        depth = int(options.pop("hops", depth))
        limit = int(options.pop("limit", 100))
        if depth < 1:
            return []
        node = self.get_node(node_id)
        if node is None:
            return []
        direction = direction.lower()
        if direction not in {"in", "out", "both"}:
            raise ValidationError(
                "HugeGraph traversal direction must be in, out or both",
                field="direction",
                value=direction,
            )
        start = self._gremlin_literal(node["id"])
        if rel_type:
            logical = self._gremlin_literal(rel_type)
            edge_filter = (
                f"or(__.hasLabel({logical}),"
                f"__.has('{self.EDGE_TYPE_PROPERTY}',{logical}))"
            )
            edge_step = {
                "out": f"outE().{edge_filter}.inV()",
                "in": f"inE().{edge_filter}.outV()",
                "both": f"bothE().{edge_filter}.otherV()",
            }[direction]
        else:
            edge_step = {"out": "out()", "in": "in()", "both": "both()"}[direction]
        gremlin = (
            f"g.V({start}).repeat(__.{edge_step}).emit().times({depth})"
            f".dedup().limit({max(1, limit)}).elementMap()"
        )
        raw = self._manager("gremlin").exec(gremlin)
        results: List[Dict[str, Any]] = []
        for value in self._gremlin_data(raw):
            neighbor = self._normalize_element_map(value)
            if not neighbor:
                continue
            props = neighbor["properties"]
            content = next(
                (
                    str(props.get(name))
                    for name in self.content_properties
                    if props.get(name) is not None
                ),
                "",
            )
            results.append(
                {
                    "id": neighbor["id"],
                    "type": neighbor["type"],
                    "content": content,
                    "metadata": props,
                    "properties": props,
                    "labels": neighbor["labels"],
                }
            )
        return results

    def shortest_path(
        self,
        start_node_id: NodeId,
        end_node_id: NodeId,
        rel_type: Optional[str] = None,
        max_depth: int = 10,
        **options: Any,
    ) -> Optional[Dict[str, Any]]:
        source = self.get_node(start_node_id)
        target = self.get_node(end_node_id)
        if source is None or target is None:
            return None
        if rel_type is None:
            traverser = self._manager("traverser")
            method = getattr(traverser, "shortest_path", None) or getattr(
                traverser, "shortestPath", None
            )
            if callable(method):
                try:
                    raw = method(source["id"], target["id"], max_depth)
                    normalized = self._normalize_path_response(raw)
                    if normalized:
                        return normalized
                except (TypeError, AttributeError):
                    # SDK signatures differ slightly by release; Gremlin below
                    # remains a native HugeGraph fallback.
                    pass
        source_literal = self._gremlin_literal(source["id"])
        target_literal = self._gremlin_literal(target["id"])
        if rel_type:
            logical = self._gremlin_literal(rel_type)
            edge_filter = (
                f"or(__.hasLabel({logical}),"
                f"__.has('{self.EDGE_TYPE_PROPERTY}',{logical}))"
            )
            step = f"bothE().{edge_filter}.otherV().simplePath()"
        else:
            step = "both().simplePath()"
        gremlin = (
            f"g.V({source_literal}).repeat(__.{step})"
            f".emit(__.hasId({target_literal})).times({int(max_depth)})"
            f".hasId({target_literal}).path().limit(1)"
        )
        raw = self._manager("gremlin").exec(gremlin)
        paths = self._gremlin_data(raw)
        return self._normalize_path_response(paths[0]) if paths else None

    def _normalize_path_response(self, raw: Any) -> Optional[Dict[str, Any]]:
        if raw is None:
            return None
        if isinstance(raw, Mapping):
            if "path" in raw:
                sequence = raw["path"]
            elif "objects" in raw:
                # GraphSON Path responses use {labels, objects}.
                sequence = raw["objects"]
            elif "vertices" in raw:
                vertex_ids = raw.get("vertices") or []
                edge_ids = raw.get("edges") or []
                nodes = []
                for vertex_id in vertex_ids:
                    node = self.get_node(vertex_id)
                    if node is not None:
                        nodes.append(node)
                graph = self._manager("graph")
                relationships = []
                for edge_id in edge_ids:
                    try:
                        relationships.append(
                            self._normalize_edge(graph.getEdgeById(edge_id))
                        )
                    except Exception:
                        continue
                return {
                    "length": max(0, len(nodes) - 1),
                    "nodes": nodes,
                    "relationships": relationships,
                }
            else:
                sequence = list(raw.values())
        else:
            sequence = raw
        if not isinstance(sequence, (list, tuple)):
            return None
        nodes: List[Dict[str, Any]] = []
        relationships: List[Dict[str, Any]] = []
        for item in sequence:
            if isinstance(item, Mapping) and any(
                key in item
                for key in ("outV", "inV", "source", "target", "OUT", "IN")
            ):
                relationships.append(self._normalize_edge(item))
            elif isinstance(item, Mapping):
                if "properties" in item and any(
                    key in item for key in ("id", "~id", "T.id")
                ):
                    node = self._normalize_vertex(item)
                else:
                    node = self._normalize_element_map(item)
                if node:
                    nodes.append(node)
            else:
                node = self.get_node(item)
                if node:
                    nodes.append(node)
        return {
            "length": max(0, len(nodes) - 1),
            "nodes": nodes,
            "relationships": relationships,
        }

    def k_out(
        self,
        source: NodeId,
        depth: int,
        direction: str = "BOTH",
        edge_steps: Optional[List[Dict[str, Any]]] = None,
        nearest: bool = False,
        capacity: int = 10_000_000,
        limit: int = 100,
    ) -> Any:
        """Call HugeGraph's native K-out Traverser API."""
        node = self.get_node(source)
        if node is None:
            return []
        traverser = self._manager("traverser")
        method = getattr(traverser, "k_out", None) or getattr(traverser, "kOut", None)
        if not callable(method):
            raise ProcessingError("Installed pyhugegraph has no K-out Traverser API")
        try:
            # Current pyhugegraph exposes the compact REST form.
            return method(node["id"], int(depth))
        except TypeError:
            # Keep compatibility with clients that expose the richer request.
            return method(
                node["id"],
                direction,
                edge_steps or [],
                int(depth),
                bool(nearest),
                int(capacity),
                int(limit),
            )

    def load_subgraph(
        self,
        seed_ids: Sequence[NodeId],
        depth: int = 1,
        direction: str = "both",
        rel_type: Optional[str] = None,
        limit: int = 500,
    ) -> Dict[str, List[Dict[str, Any]]]:
        nodes: Dict[Any, Dict[str, Any]] = {}
        for seed_id in seed_ids:
            seed = self.get_node(seed_id)
            if seed:
                nodes[seed["id"]] = seed
            for neighbor in self.get_neighbors(
                seed_id,
                rel_type=rel_type,
                direction=direction,
                depth=depth,
                limit=max(1, limit - len(nodes)),
            ):
                full = self.get_node(neighbor["id"])
                nodes[neighbor["id"]] = full or neighbor
                if len(nodes) >= limit:
                    break
            if len(nodes) >= limit:
                break
        relationships: Dict[Any, Dict[str, Any]] = {}
        for node_id in list(nodes):
            for edge in self.get_relationships(
                node_id=node_id,
                rel_type=rel_type,
                direction="both",
                limit=limit,
            ):
                if (
                    edge["start_node_id"] in nodes
                    and edge["end_node_id"] in nodes
                ):
                    relationships[edge["id"]] = edge
                if len(relationships) >= limit:
                    break
        return {
            "nodes": list(nodes.values()),
            "relationships": list(relationships.values()),
        }

    def create_index(
        self,
        label: str,
        property_name: str,
        index_type: str = "btree",
        **options: Any,
    ) -> bool:
        physical_label = self._schema_name(label, prefix="vertex")
        physical_property = self._property_name(property_name)
        existing_label = self._get_schema_item("vertexlabels", physical_label)
        if existing_label is None:
            raise ValidationError(
                f"HugeGraph vertex label '{physical_label}' does not exist",
                field="label",
                value=label,
            )
        index_kind = index_type.lower()
        method_names = {
            "btree": ("secondary",),
            "secondary": ("secondary",),
            "range": ("range", "range_index"),
            "search": ("search",),
            "fulltext": ("search",),
            "shard": ("shard",),
            "unique": ("unique",),
        }
        if index_kind not in method_names:
            raise ValidationError(
                "Unsupported HugeGraph index type",
                field="index_type",
                value=index_type,
            )
        digest = hashlib.sha1(  # nosec B324
            f"{physical_label}:{physical_property}:{index_kind}".encode("utf-8")
        ).hexdigest()[:8]
        index_name = self._schema_name(
            f"idx_{physical_label}_{physical_property}_{digest}", prefix="idx"
        )
        schema = self._manager("schema")
        try:
            if schema.getIndexLabel(index_name) is not None:
                return True
        except Exception:
            pass
        builder = schema.indexLabel(index_name)
        builder = self._builder_call(builder, ("onV", "on_v"), physical_label)
        builder = builder.by(physical_property)
        builder = self._builder_call(builder, method_names[index_kind])
        if_not_exist = getattr(builder, "ifNotExist", None) or getattr(
            builder, "if_not_exist", None
        )
        if callable(if_not_exist):
            builder = if_not_exist()
        builder.create()
        return True

    def get_stats(self) -> Dict[str, Any]:
        raw = self._manager("gremlin").exec(
            "[vertices:g.V().count().next(),edges:g.E().count().next(),"
            "vertex_labels:g.V().groupCount().by(label).next(),"
            "edge_labels:g.E().groupCount().by(label).next()]"
        )
        data = self._gremlin_data(raw)
        stats = data[0] if data and isinstance(data[0], Mapping) else {}
        return {
            "backend": "hugegraph",
            "graph": self.graph_name,
            "graphspace": self.graphspace,
            "node_count": stats.get("vertices", 0),
            "relationship_count": stats.get("edges", 0),
            "node_labels": stats.get("vertex_labels", {}),
            "relationship_types": stats.get("edge_labels", {}),
        }

    # Semantica Reasoner compatibility aliases.
    assert_fact = add_fact
    remove_fact = retract_fact
    discard_fact = retract_fact
