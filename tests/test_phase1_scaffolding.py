import pytest
from shared.embeddings.local import LocalSentenceTransformerEmbeddings
from shared.flower.messages import QueryIns, QueryRes, RegisterIns
from shared.flower.serialization import FlowerMessageSerializer
from shared.schemas.document import SearchResult


def test_embedding_pipeline():
    embedder = LocalSentenceTransformerEmbeddings()
    vector = embedder.embed_query("Sample federal query")
    assert len(vector) == embedder.dimension
    assert embedder.dimension == 384


def test_query_serialization_roundtrip():
    original_ins = QueryIns(query="What is TASR?", top_k=3, filters={"domain": "cs"})
    recordset = FlowerMessageSerializer.query_ins_to_recordset(original_ins)
    restored_ins = FlowerMessageSerializer.recordset_to_query_ins(recordset)

    assert restored_ins.query == original_ins.query
    assert restored_ins.top_k == original_ins.top_k
    assert restored_ins.filters == original_ins.filters


def test_query_res_serialization_with_arrays():
    mock_results = [SearchResult(content="TASR paper", score=0.92, metadata={"id": "doc_1"})]
    mock_embeddings = [[0.1] * 384, [0.2] * 384]

    original_res = QueryRes(
        results=mock_results,
        node_id="node-1",
        processing_time=0.045,
        trust_score=0.98,
        doc_embeddings=mock_embeddings,
    )

    recordset = FlowerMessageSerializer.query_res_to_recordset(original_res)
    restored_res = FlowerMessageSerializer.recordset_to_query_res(recordset)

    assert restored_res.node_id == "node-1"
    assert len(restored_res.results) == 1
    assert restored_res.results[0].content == "TASR paper"
    assert len(restored_res.doc_embeddings) == 2
    assert len(restored_res.doc_embeddings[0]) == 384