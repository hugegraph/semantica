"""Native Apache HugeGraph backend for Semantica.

The public store composes focused internal implementations for strong schema,
CRUD, GraphRAG/query/traversal, and native server algorithms. It uses
pyhugegraph, Gremlin, Traverser APIs and Algorithm Jobs directly; Cypher is
never translated.
"""

from __future__ import annotations

import re
from typing import Any, Dict, Optional, Sequence, Tuple, Union

from ..utils.exceptions import ProcessingError, ValidationError
from ..utils.logging import get_logger
from .hugegraph_algorithms import HugeGraphAlgorithmMixin
from .hugegraph_crud import HugeGraphCrudMixin
from .hugegraph_query import HugeGraphQueryMixin
from .hugegraph_schema import HugeGraphSchemaMixin

try:  # Optional dependency installed from requirements-hugegraph.txt
    from pyhugegraph.client import PyHugeClient
except ImportError:  # pragma: no cover - exercised by optional-dependency test
    PyHugeClient = None  # type: ignore[assignment]


class HugeGraphStore(
    HugeGraphAlgorithmMixin,
    HugeGraphQueryMixin,
    HugeGraphCrudMixin,
    HugeGraphSchemaMixin,
):
    """Semantica graph-store backend backed by Apache HugeGraph.

    ``schema_mode='auto'`` creates and extends backend-managed schema.
    ``schema_mode='strict'`` reads an existing managed graph without changing
    its schema and rejects writes that do not match it.
    """

    RESERVED_PREFIX = "_semantica_"
    LABELS_PROPERTY = "_semantica_labels"
    EDGE_TYPE_PROPERTY = "_semantica_type"
    EDGE_KEY_PROPERTY = "_semantica_edge_key"
    FACTS_PROPERTY = "_semantica_facts"
    FACT_OWNED_PROPERTY = "_semantica_fact_owned"
    APP_ID_PROPERTY = "id"

    TERMINAL_TASK_STATES = {
        "SUCCESS",
        "FAILED",
        "CANCELLED",
        "CANCELED",
        "DELETED",
    }
    SUCCESS_TASK_STATES = {"SUCCESS"}

    ALGORITHM_ALIASES: Dict[str, str] = {
        "pagerank": "page_rank",
        "page_rank": "page_rank",
        "louvain": "louvain",
        "lpa": "lpa",
        "label_propagation": "lpa",
        "weak_connected_component": "weak_connected_component",
        "weakly_connected_components": "weak_connected_component",
        "connected_components": "weak_connected_component",
        "degree": "degree_centrality",
        "degree_centrality": "degree_centrality",
        "betweenness": "betweenness_centrality",
        "betweenness_centrality": "betweenness_centrality",
        "betweenness_centrality_v2": "betweenness_centrality_v2",
        "closeness": "closeness_centrality",
        "closeness_centrality": "closeness_centrality",
        "closeness_centrality_v2": "closeness_centrality_v2",
        "stress": "stress_centrality",
        "stress_centrality": "stress_centrality",
        "stress_centrality_v2": "stress_centrality_v2",
        "eigenvector": "eigenvector_centrality",
        "eigenvector_centrality": "eigenvector_centrality",
        "k_core": "k_core",
        "triangle_count": "triangle_count",
        "clustering_coefficient": "cluster_coefficient",
        "cluster_coefficient": "cluster_coefficient",
        "rings": "rings_detect",
        "rings_detect": "rings_detect",
        "count_vertex": "count_vertex",
        "count_edge": "count_edge",
        "subgraph_stat": "subgraph_stat",
    }

    _SAFE_SCHEMA_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
    _FACT_RE = re.compile(r"^\s*([^()\s]+)\s*\((.*)\)\s*$")
    _TOKEN_RE = re.compile(r"[\w\-]+", re.UNICODE)
    _STOP_WORDS = {
        "a", "an", "and", "are", "as", "at", "be", "by", "for",
        "from", "how", "in", "is", "it", "of", "on", "or", "that",
        "the", "this", "to", "what", "which", "who", "why", "with",
    }

    def __init__(
        self,
        url: str = "http://127.0.0.1:8080",
        graph: str = "hugegraph",
        user: str = "admin",
        password: str = "admin",
        graphspace: Optional[str] = None,
        timeout: Optional[Union[float, Tuple[float, float]]] = None,
        schema_mode: str = "auto",
        candidate_limit: int = 400,
        content_properties: Optional[Sequence[str]] = None,
        client: Any = None,
        connect_on_init: bool = False,
        **options: Any,
    ) -> None:
        self.url = url.rstrip("/")
        self.graph_name = graph
        self.user = user
        self.password = password
        self.graphspace = graphspace
        self.timeout = timeout
        self.schema_mode = schema_mode.lower().strip()
        if self.schema_mode not in {"auto", "strict"}:
            raise ValidationError(
                "HugeGraph schema_mode must be either 'auto' or 'strict'",
                field="schema_mode",
                value=schema_mode,
            )
        if candidate_limit <= 0:
            raise ValidationError(
                "HugeGraph candidate_limit must be positive",
                field="candidate_limit",
                value=candidate_limit,
            )
        self.candidate_limit = int(candidate_limit)
        self.content_properties = tuple(
            content_properties
            or (
                "content",
                "text",
                "name",
                "title",
                "description",
                "summary",
                "source",
                "evidence",
            )
        )
        self.options = options
        self._client = client
        self._owns_client = client is None
        self._connected = client is not None
        self._managers: Dict[str, Any] = {}
        self._schema_cache: Dict[str, Dict[str, Any]] = {
            "propertykeys": {},
            "vertexlabels": {},
            "edgelabels": {},
        }
        self.logger = get_logger("hugegraph_store")
        if connect_on_init:
            self.connect()

    @property
    def client(self) -> Any:
        self._ensure_client()
        return self._client

    def connect(self, **options: Any) -> bool:
        """Create the pyhugegraph client and verify the graph endpoint."""
        if self._client is None:
            if PyHugeClient is None:
                raise ImportError(
                    "HugeGraph support requires the optional 'hugegraph-python' "
                    "package. Install requirements-hugegraph.txt, or "
                    "install hugegraph-python>=1.7.1 directly."
                )
            timeout = options.pop("timeout", self.timeout)
            kwargs: Dict[str, Any] = {}
            if self.graphspace:
                kwargs["graphspace"] = self.graphspace
            if timeout is not None:
                kwargs["timeout"] = timeout
            kwargs.update(options)
            try:
                self._client = PyHugeClient(
                    self.url,
                    self.graph_name,
                    self.user,
                    self.password,
                    **kwargs,
                )
            except TypeError:
                # Compatibility with older pyhugegraph releases that do not
                # expose graphspace/timeout in their constructor.
                self._client = PyHugeClient(
                    self.url,
                    self.graph_name,
                    self.user,
                    self.password,
                )
        try:
            self._manager("schema").getSchema()
        except Exception as exc:  # pragma: no cover - exercised in integration
            self._managers.clear()
            if self._owns_client:
                self._client = None
            self._connected = False
            raise ProcessingError(
                "Unable to connect to HugeGraph",
                processing_context={
                    "url": self.url,
                    "graph": self.graph_name,
                    "graphspace": self.graphspace,
                },
            ) from exc
        self._connected = True
        return True

    def close(self) -> None:
        """Release owned HTTP sessions without mutating externally owned clients."""
        if self._owns_client:
            seen: set[int] = set()
            for manager in self._managers.values():
                session = getattr(manager, "session", None)
                if session is None or id(session) in seen:
                    continue
                seen.add(id(session))
                close = getattr(session, "close", None)
                if callable(close):
                    close()
            # PyHugeClient caches manager/session objects. Recreate the client
            # on the next connect() rather than reusing closed cached sessions.
            self._client = None
        self._managers.clear()
        self._connected = False

    def _ensure_client(self) -> None:
        if self._client is None:
            self.connect()

    def _manager(self, name: str) -> Any:
        self._ensure_client()
        if name not in self._managers:
            factory = getattr(self._client, name, None)
            if not callable(factory):
                raise ProcessingError(
                    f"Installed pyhugegraph client does not provide {name}()",
                    processing_context={"manager": name},
                )
            self._managers[name] = factory()
        return self._managers[name]
