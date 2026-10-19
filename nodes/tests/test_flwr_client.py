#Pseudo-Testing code, 
#Currently not the part of main project



"""Tests for the Flower client query and registration handlers."""

from types import SimpleNamespace
from typing import Any

import pytest
from flwr.common import ConfigRecord, RecordDict

import nodes.app.flwr_client.client as client_module
from shared.flower.messages import QueryIns
from shared.flower.serialization import FlowerMessageSerializer
from shared.schemas.document import SearchResult


class FakeClientApp:
    """Capture the registered query handler without starting a Flower runtime."""

    def query(self):
        def register(handler):
            self.query_handler = handler
            return handler

        return register


class FakeMessage:
    def __init__(self, content: RecordDict) -> None:
        self.content = content
        self.reply_contents: list[RecordDict] = []

    def create_reply(self, content: RecordDict) -> "FakeMessage":
        self.reply_contents.append(content)
        return FakeMessage(content)


class FakeProfiler:
    def __init__(self, profile: tuple[list[float], list[list[float]]]) -> None:
        self.profile = profile
        self.compute_profile_calls = 0

    def compute_profile(self) -> tuple[list[float], list[list[float]]]:
        self.compute_profile_calls += 1
        return self.profile


class FakeChromaStore:
    def __init__(
        self,
        results: list[SearchResult] | None = None,
        embeddings: list[list[float]] | None = None,
        error: Exception | None = None,
    ) -> None:
        self.results = results if results is not None else []
        self.embeddings = embeddings if embeddings is not None else []
        self.error = error
        self.query_calls: list[dict[str, Any]] = []

    def query(self, **kwargs: Any) -> tuple[list[SearchResult], list[list[float]]]:
        self.query_calls.append(kwargs)
        if self.error is not None:
            raise self.error
        return self.results, self.embeddings


@pytest.fixture
def client_factory(monkeypatch):
    monkeypatch.setattr(client_module, "ClientApp", FakeClientApp)

    def create(
        chroma_store: FakeChromaStore | None = None,
        profiler: FakeProfiler | None = None,
        node_id: str = "node-7",
        domain: str = "finance",
    ):
        app = client_module.create_node_client_app(
            node_id=node_id,
            domain=domain,
            chroma_store=chroma_store or FakeChromaStore(),
            profiler=profiler or FakeProfiler(([], [])),
        )
        return app.query_handler

    return create


def make_registration_request(action: str = "register") -> RecordDict:
    request = RecordDict()
    request.configs_records["query_ins"] = ConfigRecord({"action": action})
    return request


def make_query_request(
    query: str = "finance policy",
    top_k: int = 3,
    filters: dict[str, Any] | None = None,
) -> RecordDict:
    return FlowerMessageSerializer.query_ins_to_recordset(
        QueryIns(query=query, top_k=top_k, filters=filters or {})
    )


def test_register_request_serializes_node_identity_and_profile(client_factory):
    profile = ([0.1, 0.2], [[0.3, 0.4], [0.5, 0.6]])
    profiler = FakeProfiler(profile)
    handler = client_factory(
        profiler=profiler,
        node_id="finance-node",
        domain="finance",
    )
    request = FakeMessage(make_registration_request())

    reply = handler(request, SimpleNamespace())
    registration = FlowerMessageSerializer.recordset_to_register_ins(reply.content)

    assert len(request.reply_contents) == 1
    assert profiler.compute_profile_calls == 1
    assert registration.node_id == "finance-node"
    assert registration.address == "finance-node:9091"
    assert registration.domain == "finance"
    assert registration.capabilities == ["chromadb", "retrieval"]
    assert registration.centroid == pytest.approx(profile[0])
    for actual, expected in zip(registration.profile_centroids, profile[1]):
        assert actual == pytest.approx(expected)


def test_register_request_handles_empty_profile(client_factory):
    profiler = FakeProfiler(([0.0, 0.0], []))
    handler = client_factory(profiler=profiler)

    reply = handler(FakeMessage(make_registration_request()), SimpleNamespace())
    registration = FlowerMessageSerializer.recordset_to_register_ins(reply.content)

    assert registration.centroid == [0.0, 0.0]
    assert registration.profile_centroids == []
    assert profiler.compute_profile_calls == 1


def test_register_request_propagates_profiler_errors(client_factory):
    profiler_error = RuntimeError("profile unavailable")

    class FailingProfiler:
        def compute_profile(self):
            raise profiler_error

    handler = client_factory(profiler=FailingProfiler())

    with pytest.raises(RuntimeError, match="profile unavailable"):
        handler(FakeMessage(make_registration_request()), SimpleNamespace())


def test_query_request_passes_query_parameters_and_serializes_response(
    client_factory, monkeypatch
):
    results = [
        SearchResult(
            content="A finance document",
            score=0.9,
            metadata={"id": "doc-1", "domain": "finance"},
        ),
        SearchResult(
            content="A second document",
            score=0.75,
            metadata={"id": "doc-2"},
        ),
    ]
    embeddings = [[0.25, 0.75], [0.5, 0.5]]
    store = FakeChromaStore(results=results, embeddings=embeddings)
    handler = client_factory(chroma_store=store)
    request = FakeMessage(
        make_query_request(
            query="finance policy",
            top_k=3,
            filters={"domain": "finance"},
        )
    )
    times = iter([10.0, 10.25])
    monkeypatch.setattr(client_module.time, "time", lambda: next(times))

    reply = handler(request, SimpleNamespace())
    query_result = FlowerMessageSerializer.recordset_to_query_res(reply.content)

    assert len(request.reply_contents) == 1
    assert store.query_calls == [
        {
            "query_text": "finance policy",
            "top_k": 3,
            "filters": {"domain": "finance"},
        }
    ]
    assert query_result.node_id == "node-7"
    assert query_result.processing_time == pytest.approx(0.25)
    assert query_result.trust_score == 1.0
    assert [result.model_dump() for result in query_result.results] == [
        result.model_dump() for result in results
    ]
    assert query_result.doc_embeddings == embeddings


def test_query_request_passes_none_for_empty_filters(client_factory):
    store = FakeChromaStore()
    handler = client_factory(chroma_store=store)

    handler(
        FakeMessage(make_query_request(query="plain query", top_k=1)),
        SimpleNamespace(),
    )

    assert store.query_calls == [
        {"query_text": "plain query", "top_k": 1, "filters": None}
    ]


def test_query_request_serializes_empty_results_without_embeddings(
    client_factory,
):
    store = FakeChromaStore()
    handler = client_factory(chroma_store=store)

    reply = handler(
        FakeMessage(make_query_request()),
        SimpleNamespace(),
    )
    query_result = FlowerMessageSerializer.recordset_to_query_res(reply.content)

    assert query_result.results == []
    assert query_result.doc_embeddings == []
    assert query_result.node_id == "node-7"
    assert query_result.trust_score == 1.0


def test_query_request_propagates_store_errors(client_factory):
    store_error = RuntimeError("vector store unavailable")
    handler = client_factory(chroma_store=FakeChromaStore(error=store_error))

    with pytest.raises(RuntimeError, match="vector store unavailable"):
        handler(
            FakeMessage(make_query_request()),
            SimpleNamespace(),
        )


def test_unrecognized_action_follows_query_handler(client_factory):
    store = FakeChromaStore()
    handler = client_factory(chroma_store=store)
    request = RecordDict()
    request.configs_records["query_ins"] = ConfigRecord(
        {
            "action": "unknown",
            "query": "fallback query",
            "top_k": 2,
            "filters": "{}",
        }
    )

    handler(FakeMessage(request), SimpleNamespace())

    assert store.query_calls == [
        {"query_text": "fallback query", "top_k": 2, "filters": None}
    ]
