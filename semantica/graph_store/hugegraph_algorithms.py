"""HugeGraph Traverser and standalone server Algorithm Job integration."""

from __future__ import annotations

import json
import time
from typing import Any, Dict, List, Mapping, Optional, Union

from ..utils.exceptions import ProcessingError


NodeId = Union[int, str]


class HugeGraphAlgorithmMixin:
    """Internal implementation mixin; composed by :class:`HugeGraphStore`."""

    def personal_rank(
        self,
        source: NodeId,
        label: str,
        alpha: float = 0.85,
        max_degree: int = 10_000,
        max_depth: int = 5,
        limit: int = 100,
        max_diff: float = 0.0001,
        sorted: bool = True,
        with_label: str = "BOTH_LABEL",
    ) -> Dict[str, float]:
        """Call HugeGraph's synchronous native PersonalRank Traverser."""
        node = self.get_node(source)
        if node is None:
            raise ValueError(f"Node with ID {source!r} not found")
        payload = {
            "source": node["id"],
            "label": label,
            "alpha": alpha,
            "max_degree": max_degree,
            "max_depth": max_depth,
            "limit": limit,
            "max_diff": max_diff,
            "sorted": sorted,
            "with_label": with_label,
        }
        rank_manager = None
        try:
            rank_manager = self._manager("rank")
        except ProcessingError:
            rank_manager = None
        if rank_manager is not None:
            method = getattr(rank_manager, "personal_rank", None) or getattr(
                rank_manager, "personalRank", None
            )
            if callable(method):
                try:
                    from pyhugegraph.structure.rank_data import PersonalRankParameters

                    result = method(PersonalRankParameters(**payload))
                    return dict(result or {})
                except (ImportError, TypeError):
                    pass
        result = self._raw_request("traversers/personalrank", "POST", payload)
        return dict(result or {})

    def submit_algorithm(
        self,
        name: str,
        parameters: Optional[Dict[str, Any]] = None,
        *,
        wait: bool = False,
        timeout: float = 300.0,
        poll_interval: float = 0.5,
    ) -> Dict[str, Any]:
        """Schedule a HugeGraph server-side single-node OLAP algorithm job."""
        normalized = self.ALGORITHM_ALIASES.get(name.lower(), name.lower())
        response = self._raw_request(
            f"jobs/algorithm/{normalized}", "POST", parameters or {}
        )
        task_id = (
            response.get("task_id")
            if isinstance(response, Mapping)
            else response
        )
        if task_id is None:
            raise ProcessingError(
                "HugeGraph algorithm API did not return task_id",
                processing_context={"algorithm": normalized, "response": response},
            )
        submitted = {
            "algorithm": normalized,
            "task_id": task_id,
            "status": "SUBMITTED",
        }
        if not wait:
            return submitted
        completed = self.wait_for_algorithm(
            task_id, timeout=timeout, poll_interval=poll_interval
        )
        completed.setdefault("algorithm", normalized)
        return completed

    def get_algorithm_task(self, task_id: Union[int, str]) -> Dict[str, Any]:
        manager = self._manager("task")
        method = getattr(manager, "get_task", None) or getattr(
            manager, "getTask", None
        )
        if not callable(method):
            raise ProcessingError(
                "Installed pyhugegraph client does not provide task lookup"
            )
        task = method(task_id)
        if isinstance(task, Mapping) and isinstance(task.get("task"), Mapping):
            task = task["task"]
        result = dict(task or {})
        if "result" in result:
            result["result"] = self._decode_job_result(result["result"])
        return result

    @classmethod
    def _decode_job_result(cls, value: Any) -> Any:
        current = value
        for _ in range(3):
            if not isinstance(current, str):
                break
            try:
                current = json.loads(current)
            except (TypeError, ValueError):
                break
        return current

    def wait_for_algorithm(
        self,
        task_id: Union[int, str],
        *,
        timeout: float = 300.0,
        poll_interval: float = 0.5,
    ) -> Dict[str, Any]:
        deadline = time.monotonic() + timeout
        while True:
            task = self.get_algorithm_task(task_id)
            status = str(task.get("status", "UNKNOWN")).upper()
            if status in self.TERMINAL_TASK_STATES:
                if status not in self.SUCCESS_TASK_STATES:
                    raise ProcessingError(
                        "HugeGraph algorithm job failed",
                        processing_context={
                            "task_id": task_id,
                            "status": status,
                            "task": task,
                        },
                    )
                return task
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    f"Timed out waiting for HugeGraph algorithm task {task_id}"
                )
            time.sleep(max(0.01, poll_interval))

    def run_algorithm(
        self, name: str, parameters: Optional[Dict[str, Any]] = None, **options: Any
    ) -> Dict[str, Any]:
        return self.submit_algorithm(name, parameters, **options)

    def _run_named_algorithm(
        self, name: str, parameters: Dict[str, Any]
    ) -> Dict[str, Any]:
        controls = {
            "wait": parameters.pop("wait", True),
            "timeout": parameters.pop("timeout", 300.0),
            "poll_interval": parameters.pop("poll_interval", 0.5),
        }
        return self.submit_algorithm(name, parameters, **controls)

    def page_rank(self, **parameters: Any) -> Dict[str, Any]:
        return self._run_named_algorithm("page_rank", parameters)

    def louvain(self, **parameters: Any) -> Dict[str, Any]:
        return self._run_named_algorithm("louvain", parameters)

    def label_propagation(self, **parameters: Any) -> Dict[str, Any]:
        return self._run_named_algorithm("lpa", parameters)

    def connected_components(self, **parameters: Any) -> Dict[str, Any]:
        return self._run_named_algorithm("weak_connected_component", parameters)

    def degree_centrality(self, **parameters: Any) -> Dict[str, Any]:
        return self._run_named_algorithm("degree_centrality", parameters)

    def betweenness_centrality(self, **parameters: Any) -> Dict[str, Any]:
        return self._run_named_algorithm("betweenness_centrality", parameters)

    def closeness_centrality(self, **parameters: Any) -> Dict[str, Any]:
        return self._run_named_algorithm("closeness_centrality", parameters)

    def k_core(self, **parameters: Any) -> Dict[str, Any]:
        return self._run_named_algorithm("k_core", parameters)

    def triangle_count(self, **parameters: Any) -> Dict[str, Any]:
        return self._run_named_algorithm("triangle_count", parameters)

    def clustering_coefficient(self, **parameters: Any) -> Dict[str, Any]:
        return self._run_named_algorithm("cluster_coefficient", parameters)

    def stress_centrality(self, **parameters: Any) -> Dict[str, Any]:
        return self._run_named_algorithm("stress_centrality", parameters)

    def eigenvector_centrality(self, **parameters: Any) -> Dict[str, Any]:
        return self._run_named_algorithm("eigenvector_centrality", parameters)

    @staticmethod
    def _batch_response_ids(response: Any, *, expected: int) -> List[Any]:
        ids = response
        if isinstance(response, Mapping):
            ids = response.get("ids", response.get("data"))
        if not isinstance(ids, list) or len(ids) != expected:
            raise ProcessingError(
                "HugeGraph batch API returned an unexpected ID list",
                processing_context={
                    "expected": expected,
                    "response": response,
                },
            )
        return ids

    @staticmethod
    def _endpoint_unavailable(error: BaseException) -> bool:
        """Return True only for a missing REST endpoint/session capability.

        Batch writes must not be retried as single writes after an arbitrary
        transport failure because the server might already have committed the
        transaction. Fallback is therefore limited to explicit 404/405
        responses and old client doubles that expose no native HTTP session.
        """
        current: Optional[BaseException] = error
        while current is not None:
            response = getattr(current, "response", None)
            status = getattr(response, "status_code", None)
            if status in {404, 405}:
                return True
            text = str(current).lower()
            if "does not expose its http session" in text:
                return True
            current = current.__cause__
        return False

    def _http_session(self) -> Any:
        for manager_name in ("graph", "task", "traverser", "gremlin"):
            try:
                manager = self._manager(manager_name)
            except ProcessingError:
                continue
            session = getattr(manager, "session", None)
            if session is not None:
                return session
        raise ProcessingError("pyhugegraph manager does not expose its HTTP session")

    def _raw_request(self, path: str, method: str, payload: Any = None) -> Any:
        session = self._http_session()
        kwargs: Dict[str, Any] = {}
        if payload is not None:
            kwargs["data"] = json.dumps(payload, ensure_ascii=False)
        try:
            return session.request(path, method, **kwargs)
        except Exception as exc:
            raise ProcessingError(
                "HugeGraph native REST request failed",
                processing_context={"path": path, "method": method},
            ) from exc
