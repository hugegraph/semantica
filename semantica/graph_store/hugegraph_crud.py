"""Native HugeGraph vertex and edge CRUD, including REST batch operations."""

from __future__ import annotations

import json
import uuid
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

from ..utils.exceptions import ProcessingError, ValidationError


NodeId = Union[int, str]


class HugeGraphCrudMixin:
    """Internal implementation mixin; composed by :class:`HugeGraphStore`."""

    def _normalize_vertex(self, vertex: Any) -> Dict[str, Any]:
        if vertex is None:
            return {}
        properties = dict(self._attr(vertex, "properties", default={}) or {})
        vertex_id = self._attr(vertex, "id", "~id")
        physical_label = self._attr(vertex, "label", "~label", default="Entity")
        labels_value = properties.get(self.LABELS_PROPERTY)
        if isinstance(labels_value, str):
            try:
                parsed = json.loads(labels_value)
                labels = parsed if isinstance(parsed, list) else [labels_value]
            except (TypeError, ValueError):
                labels = [labels_value]
        elif isinstance(labels_value, list):
            labels = list(labels_value)
        else:
            labels = [physical_label]
        return {
            "id": vertex_id,
            "label": physical_label,
            "labels": labels,
            "type": labels[0] if labels else physical_label,
            "properties": properties,
        }

    def _normalize_edge(self, edge: Any) -> Dict[str, Any]:
        if edge is None:
            return {}
        properties = dict(self._attr(edge, "properties", default={}) or {})
        physical_label = self._attr(edge, "label", default="relationship")
        logical_type = properties.get(self.EDGE_TYPE_PROPERTY, physical_label)
        return {
            "id": self._attr(edge, "id"),
            "type": logical_type,
            "label": physical_label,
            "start_node_id": self._attr(edge, "outV", "out_v", "source", "OUT"),
            "end_node_id": self._attr(edge, "inV", "in_v", "target", "IN"),
            "start_node_label": self._attr(edge, "outVLabel", "out_v_label"),
            "end_node_label": self._attr(edge, "inVLabel", "in_v_label"),
            "properties": properties,
        }

    def _normalize_element_map(self, value: Any) -> Dict[str, Any]:
        if not isinstance(value, Mapping):
            return {}
        raw = dict(value)
        vertex_id = raw.pop("id", raw.pop("~id", raw.pop("T.id", None)))
        label = raw.pop("label", raw.pop("~label", raw.pop("T.label", "Entity")))
        properties: Dict[str, Any] = {}
        for key, item in raw.items():
            if isinstance(key, str) and key.startswith("~"):
                continue
            if isinstance(item, list) and len(item) == 1:
                item = item[0]
            properties[str(key)] = item
        return self._normalize_vertex(
            {"id": vertex_id, "label": label, "properties": properties}
        )

    @staticmethod
    def _gremlin_data(response: Any) -> List[Any]:
        if response is None:
            return []
        if isinstance(response, list):
            return response
        if not isinstance(response, Mapping):
            return [response]
        result = response.get("result", response)
        if isinstance(result, Mapping):
            data = result.get("data", result.get("results", []))
        else:
            data = result
        if data is None:
            return []
        return data if isinstance(data, list) else [data]

    def _prepare_vertex_create(
        self,
        labels: Sequence[str],
        properties: Mapping[str, Any],
        requested_id: Optional[NodeId] = None,
    ) -> Tuple[str, Dict[str, Any], Optional[NodeId], NodeId]:
        logical_labels = self._coerce_labels(labels)
        label = self._schema_name(logical_labels[0], prefix="vertex")
        props = dict(properties or {})
        if requested_id is None:
            requested_id = props.get(self.APP_ID_PROPERTY)
        if requested_id is None:
            requested_id = str(uuid.uuid4())
        props[self.APP_ID_PROPERTY] = requested_id
        props[self.LABELS_PROPERTY] = logical_labels
        normalized, _ = self._prepare_properties(props)
        vertex_label = self._ensure_vertex_label(label, list(normalized))
        id_strategy = str(
            self._attr(vertex_label, "idStrategy", "id_strategy", default="")
        ).upper()
        explicit_id: Optional[NodeId] = requested_id
        if id_strategy == "CUSTOMIZE_NUMBER":
            try:
                explicit_id = int(requested_id)
            except (TypeError, ValueError) as exc:
                raise ValidationError(
                    "HugeGraph CUSTOMIZE_NUMBER labels require an integer ID",
                    field="id",
                    value=requested_id,
                ) from exc
        elif id_strategy == "CUSTOMIZE_STRING":
            explicit_id = str(requested_id)
        elif id_strategy not in {"CUSTOMIZE_STRING", "CUSTOMIZE_NUMBER"}:
            explicit_id = None
        return label, normalized, explicit_id, requested_id

    def create_node(
        self,
        labels: List[str],
        properties: Dict[str, Any],
        **options: Any,
    ) -> Dict[str, Any]:
        label, normalized, explicit_id, requested_id = self._prepare_vertex_create(
            labels, properties, options.get("id")
        )
        try:
            created = self._manager("graph").addVertex(label, normalized, explicit_id)
        except Exception as exc:
            raise ProcessingError(
                "Failed to create HugeGraph vertex",
                processing_context={"label": label, "id": requested_id},
            ) from exc
        result = self._normalize_vertex(created)
        if not result:
            result = self.get_node(explicit_id or requested_id) or {}
        return result

    def create_nodes(
        self, nodes: List[Dict[str, Any]], **options: Any
    ) -> List[Dict[str, Any]]:
        if not nodes:
            return []
        native_batch = bool(options.pop("native_batch", True))
        fallback_on_unsupported = bool(options.pop("fallback_on_unsupported", True))
        if not native_batch:
            return [
                self.create_node(
                    self._coerce_labels(
                        node.get("labels") or node.get("type", "Entity")
                    ),
                    dict(node.get("properties") or {}),
                    id=node.get("id"),
                    **options,
                )
                for node in nodes
            ]
        prepared: List[Tuple[str, Dict[str, Any], Optional[NodeId], NodeId]] = []
        payload: List[Dict[str, Any]] = []
        for node in nodes:
            item = self._prepare_vertex_create(
                self._coerce_labels(
                    node.get("labels") or node.get("type", "Entity")
                ),
                dict(node.get("properties") or {}),
                node.get("id"),
            )
            prepared.append(item)
            label, properties, explicit_id, _requested_id = item
            raw: Dict[str, Any] = {"label": label, "properties": properties}
            if explicit_id is not None:
                raw["id"] = explicit_id
            payload.append(raw)
        try:
            response = self._raw_request("graph/vertices/batch", "POST", payload)
        except ProcessingError as exc:
            if fallback_on_unsupported and self._endpoint_unavailable(exc):
                return self.create_nodes(
                    nodes,
                    native_batch=False,
                    fallback_on_unsupported=False,
                    **options,
                )
            raise
        ids = self._batch_response_ids(response, expected=len(payload))
        results: List[Dict[str, Any]] = []
        for vertex_id, item in zip(ids, prepared):
            label, properties, _explicit_id, _requested_id = item
            results.append(
                self._normalize_vertex(
                    {"id": vertex_id, "label": label, "properties": properties}
                )
            )
        return results

    def get_node(self, node_id: NodeId, **options: Any) -> Optional[Dict[str, Any]]:
        try:
            vertex = self._manager("graph").getVertexById(node_id)
        except Exception:
            vertex = None
        if vertex is not None:
            return self._normalize_vertex(vertex)
        matches = self.get_nodes(
            properties={self.APP_ID_PROPERTY: node_id}, limit=2, **options
        )
        return matches[0] if matches else None

    def get_nodes(
        self,
        labels: Optional[List[str]] = None,
        properties: Optional[Dict[str, Any]] = None,
        limit: int = 100,
        **options: Any,
    ) -> List[Dict[str, Any]]:
        physical_label = ""
        normalized_labels: List[str] = []
        if labels:
            normalized_labels = self._coerce_labels(labels)
            physical_label = self._schema_name(normalized_labels[0], prefix="vertex")
        normalized_properties: Dict[str, Any] = {}
        for key, value in (properties or {}).items():
            normalized_properties[self._property_name(str(key))] = self._property_spec(
                value
            ).value
        try:
            vertices = self._manager("graph").getVertexByCondition(
                label=physical_label,
                limit=max(0, int(limit)),
                properties=normalized_properties or None,
            )
        except TypeError:
            vertices = self._manager("graph").getVertexByCondition(
                physical_label, max(0, int(limit)), None, normalized_properties or None
            )
        results = [self._normalize_vertex(vertex) for vertex in (vertices or [])]
        if normalized_labels and len(normalized_labels) > 1:
            required = set(normalized_labels)
            results = [
                item
                for item in results
                if required.issubset(set(item["labels"]))
            ]
        return results[:limit] if limit > 0 else results

    def update_node(
        self,
        node_id: NodeId,
        properties: Dict[str, Any],
        merge: bool = True,
        **options: Any,
    ) -> Dict[str, Any]:
        current = self.get_node(node_id)
        if current is None:
            raise ValueError(f"Node with ID {node_id!r} not found")
        normalized, _ = self._prepare_properties(properties)
        label = current.get("label") or self._schema_name(
            current["labels"][0], prefix="vertex"
        )
        self._ensure_vertex_label(label, list(normalized))
        graph = self._manager("graph")
        internal_id = current["id"]
        try:
            if not merge:
                removable = {
                    key: value
                    for key, value in current["properties"].items()
                    if key not in {self.APP_ID_PROPERTY, self.LABELS_PROPERTY}
                }
                if removable:
                    graph.eliminateVertex(internal_id, removable)
            updated = graph.appendVertex(internal_id, normalized)
        except Exception as exc:
            raise ProcessingError(
                "Failed to update HugeGraph vertex",
                processing_context={"id": node_id},
            ) from exc
        return self._normalize_vertex(updated) or self.get_node(internal_id) or {}

    def delete_node(
        self, node_id: NodeId, detach: bool = True, **options: Any
    ) -> bool:
        current = self.get_node(node_id)
        if current is None:
            return False
        internal_id = current["id"]
        if not detach and self.get_relationships(node_id=internal_id, limit=1):
            raise ValidationError(
                "Cannot delete a HugeGraph vertex with incident edges "
                "when detach=False",
                field="detach",
                value=False,
            )
        try:
            self._manager("graph").removeVertexById(internal_id)
        except Exception as exc:
            raise ProcessingError(
                "Failed to delete HugeGraph vertex",
                processing_context={"id": node_id},
            ) from exc
        return True

    def _prepare_edge_create(
        self,
        start_node_id: NodeId,
        end_node_id: NodeId,
        rel_type: str,
        properties: Optional[Mapping[str, Any]] = None,
        node_cache: Optional[Dict[NodeId, Dict[str, Any]]] = None,
    ) -> Tuple[str, Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
        cache = node_cache if node_cache is not None else {}

        def resolve(node_id: NodeId) -> Optional[Dict[str, Any]]:
            if node_id not in cache:
                node = self.get_node(node_id)
                if node is not None:
                    cache[node_id] = node
            return cache.get(node_id)

        source = resolve(start_node_id)
        target = resolve(end_node_id)
        if source is None or target is None:
            missing = start_node_id if source is None else end_node_id
            raise ValueError(f"Node with ID {missing!r} not found")
        source_label = source.get("label") or self._schema_name(
            source["labels"][0], prefix="vertex"
        )
        target_label = target.get("label") or self._schema_name(
            target["labels"][0], prefix="vertex"
        )
        physical_label = self._edge_label_name(rel_type, source_label, target_label)
        props = dict(properties or {})
        props[self.EDGE_TYPE_PROPERTY] = rel_type
        props.setdefault(self.EDGE_KEY_PROPERTY, str(uuid.uuid4()))
        normalized, _ = self._prepare_properties(props)
        self._ensure_edge_label(
            physical_label, source_label, target_label, list(normalized)
        )
        return physical_label, normalized, source, target

    def create_relationship(
        self,
        start_node_id: NodeId,
        end_node_id: NodeId,
        rel_type: str,
        properties: Optional[Dict[str, Any]] = None,
        **options: Any,
    ) -> Dict[str, Any]:
        physical_label, normalized, source, target = self._prepare_edge_create(
            start_node_id, end_node_id, rel_type, properties
        )
        try:
            edge = self._manager("graph").addEdge(
                physical_label, source["id"], target["id"], normalized
            )
        except Exception as exc:
            raise ProcessingError(
                "Failed to create HugeGraph edge",
                processing_context={
                    "type": rel_type,
                    "source": start_node_id,
                    "target": end_node_id,
                },
            ) from exc
        return self._normalize_edge(edge)

    def create_relationships(
        self, relationships: List[Dict[str, Any]], **options: Any
    ) -> List[Dict[str, Any]]:
        if not relationships:
            return []
        native_batch = bool(options.pop("native_batch", True))
        fallback_on_unsupported = bool(options.pop("fallback_on_unsupported", True))
        if not native_batch:
            return [
                self.create_relationship(
                    rel.get("start_node_id", rel.get("start_id", rel.get("source"))),
                    rel.get("end_node_id", rel.get("end_id", rel.get("target"))),
                    rel.get("type", rel.get("rel_type", "RELATED_TO")),
                    rel.get("properties"),
                    **options,
                )
                for rel in relationships
            ]
        node_cache: Dict[NodeId, Dict[str, Any]] = {}
        prepared: List[Tuple[str, Dict[str, Any], Dict[str, Any], Dict[str, Any]]] = []
        payload: List[Dict[str, Any]] = []
        for relationship in relationships:
            start_node_id = relationship.get(
                "start_node_id",
                relationship.get("start_id", relationship.get("source")),
            )
            end_node_id = relationship.get(
                "end_node_id",
                relationship.get("end_id", relationship.get("target")),
            )
            if start_node_id is None or end_node_id is None:
                raise ValidationError(
                    "HugeGraph batch edges require source and target IDs",
                    field="relationship",
                    value=relationship,
                )
            rel_type = str(
                relationship.get("type", relationship.get("rel_type", "RELATED_TO"))
            )
            item = self._prepare_edge_create(
                start_node_id,
                end_node_id,
                rel_type,
                relationship.get("properties"),
                node_cache,
            )
            prepared.append(item)
            physical_label, properties, source, target = item
            payload.append(
                {
                    "label": physical_label,
                    "outV": source["id"],
                    "inV": target["id"],
                    "outVLabel": source["label"],
                    "inVLabel": target["label"],
                    "properties": properties,
                }
            )
        try:
            response = self._raw_request("graph/edges/batch", "POST", payload)
        except ProcessingError as exc:
            if fallback_on_unsupported and self._endpoint_unavailable(exc):
                return self.create_relationships(
                    relationships,
                    native_batch=False,
                    fallback_on_unsupported=False,
                    **options,
                )
            raise
        ids = self._batch_response_ids(response, expected=len(payload))
        results: List[Dict[str, Any]] = []
        for edge_id, item in zip(ids, prepared):
            physical_label, properties, source, target = item
            results.append(
                self._normalize_edge(
                    {
                        "id": edge_id,
                        "label": physical_label,
                        "outV": source["id"],
                        "inV": target["id"],
                        "outVLabel": source["label"],
                        "inVLabel": target["label"],
                        "properties": properties,
                    }
                )
            )
        return results

    def get_relationships(
        self,
        node_id: Optional[NodeId] = None,
        rel_type: Optional[str] = None,
        direction: str = "both",
        limit: int = 100,
        **options: Any,
    ) -> List[Dict[str, Any]]:
        direction = direction.lower()
        if direction not in {"in", "out", "both"}:
            raise ValidationError(
                "HugeGraph relationship direction must be in, out or both",
                field="direction",
                value=direction,
            )
        graph = self._manager("graph")
        internal_id = None
        if node_id is not None:
            node = self.get_node(node_id)
            if node is None:
                return []
            internal_id = node["id"]
        directions = [direction.upper()] if direction != "both" else ["OUT", "IN"]
        results: List[Dict[str, Any]] = []
        seen: set[Any] = set()
        for current_direction in directions:
            remaining = max(1, limit - len(results)) if limit > 0 else 0
            try:
                edges, _ = graph.getEdgeByPage(
                    vertex_id=internal_id,
                    direction=current_direction if internal_id is not None else None,
                    limit=remaining,
                )
            except Exception as exc:
                if internal_id is None:
                    try:
                        edges, _ = graph.getEdgeByPage(limit=remaining)
                    except Exception as nested:
                        raise ProcessingError("Failed to query HugeGraph edges") from nested
                else:
                    raise ProcessingError(
                        "Failed to query HugeGraph edges",
                        processing_context={"vertex_id": node_id},
                    ) from exc
            for edge in edges or []:
                item = self._normalize_edge(edge)
                if (
                    rel_type
                    and item.get("type") != rel_type
                    and item.get("label") != rel_type
                ):
                    continue
                key = item.get("id")
                if key in seen:
                    continue
                seen.add(key)
                results.append(item)
                if limit > 0 and len(results) >= limit:
                    return results
            if internal_id is None:
                break
        return results

    def update_relationship(
        self,
        rel_id: Union[int, str],
        properties: Dict[str, Any],
        merge: bool = True,
        **options: Any,
    ) -> Dict[str, Any]:
        graph = self._manager("graph")
        try:
            current = graph.getEdgeById(rel_id)
        except Exception:
            current = None
        if current is None:
            raise ValueError(f"Relationship with ID {rel_id!r} not found")
        normalized, _ = self._prepare_properties(properties)
        current_normalized = self._normalize_edge(current)
        edge_label = current_normalized.get("label")
        source_label = current_normalized.get("start_node_label")
        target_label = current_normalized.get("end_node_label")
        if edge_label and source_label and target_label:
            self._ensure_edge_label(
                edge_label, source_label, target_label, list(normalized)
            )
        if not merge:
            existing = dict(self._attr(current, "properties", default={}) or {})
            removable = {
                key: value
                for key, value in existing.items()
                if key not in {self.EDGE_TYPE_PROPERTY, self.EDGE_KEY_PROPERTY}
            }
            if removable:
                graph.eliminateEdge(rel_id, removable)
        updated = graph.appendEdge(rel_id, normalized)
        return self._normalize_edge(updated)

    def delete_relationship(self, rel_id: Union[int, str], **options: Any) -> bool:
        try:
            self._manager("graph").removeEdgeById(rel_id)
        except Exception as exc:
            raise ProcessingError(
                "Failed to delete HugeGraph edge",
                processing_context={"id": rel_id},
            ) from exc
        return True
