"""Strong-schema mapping and value normalization for the HugeGraph backend."""

from __future__ import annotations

import base64
import datetime as _datetime
import hashlib
import json
import math
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from ..utils.exceptions import ValidationError


@dataclass(frozen=True)
class _PropertySpec:
    data_type: str
    cardinality: str
    value: Any


class HugeGraphSchemaMixin:
    """Internal implementation mixin; composed by :class:`HugeGraphStore`."""

    @classmethod
    def _schema_name(cls, value: str, *, prefix: str = "s") -> str:
        value = str(value or "").strip()
        if cls._SAFE_SCHEMA_NAME.fullmatch(value):
            return value
        cleaned = re.sub(r"[^A-Za-z0-9_]", "_", value)
        cleaned = re.sub(r"_+", "_", cleaned).strip("_") or prefix
        if cleaned[0].isdigit():
            cleaned = f"{prefix}_{cleaned}"
        digest = hashlib.sha1(value.encode("utf-8")).hexdigest()[:8]  # nosec B324
        return f"{cleaned[:80]}_{digest}"

    @classmethod
    def _property_name(cls, value: str) -> str:
        return cls._schema_name(value, prefix="p")

    @staticmethod
    def _coerce_labels(labels: Any) -> List[str]:
        if labels is None:
            return ["Entity"]
        if isinstance(labels, str):
            labels = [labels]
        normalized = [str(label) for label in labels if str(label).strip()]
        return normalized or ["Entity"]

    @classmethod
    def _edge_label_name(cls, logical: str, source: str, target: str) -> str:
        base = cls._schema_name(logical, prefix="edge")
        pair = f"{source}\0{target}".encode("utf-8")
        digest = hashlib.sha1(pair).hexdigest()[:10]  # nosec B324
        return f"{base[:80]}__{digest}"

    @staticmethod
    def _json_default(value: Any) -> Any:
        if isinstance(value, (_datetime.datetime, _datetime.date, _datetime.time)):
            return value.isoformat()
        if isinstance(value, bytes):
            return base64.b64encode(value).decode("ascii")
        if isinstance(value, set):
            return sorted(value, key=repr)
        return str(value)

    @classmethod
    def _canonical_json(cls, value: Any) -> str:
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            default=cls._json_default,
        )

    @classmethod
    def _single_type(cls, value: Any) -> Tuple[str, Any]:
        if isinstance(value, bool):
            return "BOOLEAN", value
        if isinstance(value, int) and not isinstance(value, bool):
            return "LONG", value
        if isinstance(value, float):
            if not math.isfinite(value):
                return "TEXT", str(value)
            return "DOUBLE", value
        if isinstance(value, (_datetime.datetime, _datetime.date, _datetime.time)):
            return "TEXT", value.isoformat()
        if isinstance(value, bytes):
            return "TEXT", base64.b64encode(value).decode("ascii")
        if isinstance(value, str):
            return "TEXT", value
        return "TEXT", cls._canonical_json(value)

    @classmethod
    def _property_spec(cls, value: Any) -> _PropertySpec:
        if isinstance(value, (list, tuple, set)):
            values = list(value)
            if not values:
                return _PropertySpec("TEXT", "LIST", [])
            typed = [cls._single_type(item) for item in values]
            data_types = {item[0] for item in typed}
            if len(data_types) == 1 and data_types <= {
                "BOOLEAN",
                "LONG",
                "DOUBLE",
                "TEXT",
            }:
                return _PropertySpec(typed[0][0], "LIST", [item[1] for item in typed])
            return _PropertySpec("TEXT", "SINGLE", cls._canonical_json(values))
        data_type, normalized = cls._single_type(value)
        return _PropertySpec(data_type, "SINGLE", normalized)

    @staticmethod
    def _attr(obj: Any, *names: str, default: Any = None) -> Any:
        if obj is None:
            return default
        if isinstance(obj, Mapping):
            for name in names:
                if name in obj:
                    return obj[name]
            return default
        for name in names:
            if hasattr(obj, name):
                return getattr(obj, name)
        return default

    def _refresh_schema_cache(self) -> None:
        schema = self._manager("schema")
        self._schema_cache["propertykeys"] = {
            item.name: item for item in (schema.getPropertyKeys() or [])
        }
        self._schema_cache["vertexlabels"] = {
            item.name: item for item in (schema.getVertexLabels() or [])
        }
        self._schema_cache["edgelabels"] = {
            item.name: item for item in (schema.getEdgeLabels() or [])
        }

    def _get_schema_item(self, kind: str, name: str) -> Any:
        cache = self._schema_cache[kind]
        if name in cache:
            return cache[name]
        schema = self._manager("schema")
        method_name = {
            "propertykeys": "getPropertyKey",
            "vertexlabels": "getVertexLabel",
            "edgelabels": "getEdgeLabel",
        }[kind]
        try:
            item = getattr(schema, method_name)(name)
        except Exception:
            item = None
        if item is not None:
            cache[name] = item
        return item

    @staticmethod
    def _builder_call(builder: Any, candidates: Sequence[str], *args: Any) -> Any:
        for name in candidates:
            method = getattr(builder, name, None)
            if callable(method):
                return method(*args)
        raise AttributeError(
            f"Builder does not implement any of: {', '.join(candidates)}"
        )

    def _ensure_property_key(self, name: str, spec: _PropertySpec) -> Any:
        existing = self._get_schema_item("propertykeys", name)
        if existing is not None:
            existing_type = str(
                self._attr(existing, "dataType", "data_type", default=spec.data_type)
            ).upper()
            existing_cardinality = str(
                self._attr(existing, "cardinality", default=spec.cardinality)
            ).upper()
            if (
                existing_cardinality == "SINGLE"
                and spec.cardinality in {"LIST", "SET"}
            ):
                if existing_type != "TEXT":
                    raise ValidationError(
                        f"HugeGraph property key '{name}' is SINGLE "
                        "but received a list",
                        field=name,
                        expected=existing_cardinality,
                        actual=spec.cardinality,
                    )
                coerced = self._canonical_json(spec.value)
            else:
                try:
                    coerced = self._coerce_existing_value(
                        spec.value, existing_type, existing_cardinality
                    )
                except (TypeError, ValueError) as exc:
                    raise ValidationError(
                        f"Value is incompatible with HugeGraph property key '{name}'",
                        field=name,
                        expected=f"{existing_type}/{existing_cardinality}",
                        value=spec.value,
                    ) from exc
            return _PropertySpec(
                existing_type,
                existing_cardinality,
                coerced,
            )
        if self.schema_mode == "strict":
            raise ValidationError(
                f"HugeGraph property key '{name}' does not exist",
                field=name,
                constraint="existing HugeGraph schema",
            )
        builder = self._manager("schema").propertyKey(name)
        type_method = {
            "BOOLEAN": ("asBool", "as_bool"),
            "LONG": ("asLong", "as_long"),
            "DOUBLE": ("asDouble", "as_double"),
            "TEXT": ("asText", "as_text"),
        }.get(spec.data_type, ("asText", "as_text"))
        builder = self._builder_call(builder, type_method)
        card_method = (
            ("valueList", "value_list")
            if spec.cardinality == "LIST"
            else ("valueSingle", "value_single")
        )
        builder = self._builder_call(builder, card_method)
        if_not_exist = getattr(builder, "ifNotExist", None) or getattr(
            builder, "if_not_exist", None
        )
        if callable(if_not_exist):
            builder = if_not_exist()
        builder.create()
        self._schema_cache["propertykeys"].pop(name, None)
        return spec

    @classmethod
    def _coerce_existing_value(
        cls, value: Any, data_type: str, cardinality: str
    ) -> Any:
        values = value if isinstance(value, list) else [value]

        def coerce(item: Any) -> Any:
            if data_type in {"BYTE", "INT", "LONG"}:
                return int(item)
            if data_type in {"FLOAT", "DOUBLE"}:
                return float(item)
            if data_type == "BOOLEAN":
                if isinstance(item, str):
                    return item.strip().lower() in {"1", "true", "yes", "on"}
                return bool(item)
            if data_type == "BLOB" and isinstance(item, bytes):
                return base64.b64encode(item).decode("ascii")
            if data_type == "DATE" and isinstance(
                item, (_datetime.datetime, _datetime.date)
            ):
                return item.isoformat()
            if isinstance(item, str):
                return item
            if isinstance(item, (dict, list)):
                return cls._canonical_json(item)
            return str(item)

        coerced = [coerce(item) for item in values]
        if cardinality in {"LIST", "SET"}:
            if cardinality == "SET":
                unique: List[Any] = []
                for item in coerced:
                    if item not in unique:
                        unique.append(item)
                return unique
            return coerced
        return coerced[0] if coerced else None

    def _prepare_properties(
        self, properties: Optional[Mapping[str, Any]]
    ) -> Tuple[Dict[str, Any], Dict[str, _PropertySpec]]:
        normalized: Dict[str, Any] = {}
        specs: Dict[str, _PropertySpec] = {}
        for original_key, value in (properties or {}).items():
            if value is None:
                continue
            key = self._property_name(str(original_key))
            spec = self._property_spec(value)
            effective = self._ensure_property_key(key, spec)
            specs[key] = effective
            normalized[key] = effective.value
        return normalized, specs

    def _ensure_vertex_label(self, label: str, property_names: Sequence[str]) -> Any:
        existing = self._get_schema_item("vertexlabels", label)
        if existing is None:
            if self.schema_mode == "strict":
                raise ValidationError(
                    f"HugeGraph vertex label '{label}' does not exist",
                    field=label,
                    constraint="existing HugeGraph schema",
                )
            builder = self._manager("schema").vertexLabel(label)
            builder = self._builder_call(
                builder, ("useCustomizeStringId", "use_customize_string_id")
            )
            if property_names:
                builder = builder.properties(*property_names)
                builder = self._builder_call(
                    builder, ("nullableKeys", "nullable_keys"), *property_names
                )
            enable_index = getattr(builder, "enableLabelIndex", None) or getattr(
                builder, "enable_label_index", None
            )
            if callable(enable_index):
                builder = enable_index(True)
            if_not_exist = getattr(builder, "ifNotExist", None) or getattr(
                builder, "if_not_exist", None
            )
            if callable(if_not_exist):
                builder = if_not_exist()
            builder.create()
            self._schema_cache["vertexlabels"].pop(label, None)
            return self._get_schema_item("vertexlabels", label)

        existing_properties = set(
            self._attr(existing, "properties", default=[]) or []
        )
        missing = [name for name in property_names if name not in existing_properties]
        if missing:
            if self.schema_mode == "strict":
                raise ValidationError(
                    f"HugeGraph vertex label '{label}' is missing properties {missing}",
                    field=label,
                    constraint="existing HugeGraph schema",
                )
            builder = self._manager("schema").vertexLabel(label).properties(*missing)
            builder = self._builder_call(
                builder, ("nullableKeys", "nullable_keys"), *missing
            )
            builder.append()
            self._schema_cache["vertexlabels"].pop(label, None)
            existing = self._get_schema_item("vertexlabels", label)
        return existing

    def _ensure_edge_label(
        self,
        label: str,
        source_label: str,
        target_label: str,
        property_names: Sequence[str],
    ) -> Any:
        existing = self._get_schema_item("edgelabels", label)
        if existing is None:
            if self.schema_mode == "strict":
                raise ValidationError(
                    f"HugeGraph edge label '{label}' does not exist",
                    field=label,
                    constraint="existing HugeGraph schema",
                )
            builder = self._manager("schema").edgeLabel(label).link(
                source_label, target_label
            )
            builder = self._builder_call(builder, ("multiTimes", "multi_times"))
            if property_names:
                builder = builder.properties(*property_names)
                nullable_names = [
                    name
                    for name in property_names
                    if name != self.EDGE_KEY_PROPERTY
                ]
                if nullable_names:
                    builder = self._builder_call(
                        builder,
                        ("nullableKeys", "nullable_keys"),
                        *nullable_names,
                    )
            builder = self._builder_call(
                builder, ("sortKeys", "sort_keys"), self.EDGE_KEY_PROPERTY
            )
            if_not_exist = getattr(builder, "ifNotExist", None) or getattr(
                builder, "if_not_exist", None
            )
            if callable(if_not_exist):
                builder = if_not_exist()
            builder.create()
            self._schema_cache["edgelabels"].pop(label, None)
            return self._get_schema_item("edgelabels", label)

        actual_source = self._attr(existing, "sourceLabel", "source_label")
        actual_target = self._attr(existing, "targetLabel", "target_label")
        if actual_source and actual_source != source_label:
            raise ValidationError(
                f"HugeGraph edge label '{label}' source mismatch",
                expected=source_label,
                actual=actual_source,
            )
        if actual_target and actual_target != target_label:
            raise ValidationError(
                f"HugeGraph edge label '{label}' target mismatch",
                expected=target_label,
                actual=actual_target,
            )
        existing_properties = set(
            self._attr(existing, "properties", default=[]) or []
        )
        missing = [name for name in property_names if name not in existing_properties]
        if missing:
            if self.schema_mode == "strict":
                raise ValidationError(
                    f"HugeGraph edge label '{label}' is missing properties {missing}",
                    field=label,
                    constraint="existing HugeGraph schema",
                )
            builder = self._manager("schema").edgeLabel(label).properties(*missing)
            nullable_missing = [
                name for name in missing if name != self.EDGE_KEY_PROPERTY
            ]
            if nullable_missing:
                builder = self._builder_call(
                    builder,
                    ("nullableKeys", "nullable_keys"),
                    *nullable_missing,
                )
            builder.append()
            self._schema_cache["edgelabels"].pop(label, None)
            existing = self._get_schema_item("edgelabels", label)
        return existing
