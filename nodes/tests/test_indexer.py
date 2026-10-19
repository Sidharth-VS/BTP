#Pseudo-Testing code, 
#Currently not the part of main project


"""Tests for indexing text, Markdown, and StackExchange JSONL documents."""

import json
from pathlib import Path

import pytest

from nodes.app.rag.indexer import DocumentIndexer


class FakeChromaStore:
    def __init__(self) -> None:
        self.added_chunks = None
        self.add_chunks_calls = 0

    def add_chunks(self, chunks) -> None:
        self.add_chunks_calls += 1
        self.added_chunks = list(chunks)


def write_text(path: Path, content: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path


def test_missing_directory_returns_zero_without_writing_to_store(tmp_path):
    store = FakeChromaStore()
    indexer = DocumentIndexer(store)

    count = indexer.index_directory(tmp_path / "does-not-exist", node_id="node-1")

    assert count == 0
    assert store.add_chunks_calls == 0
    assert store.added_chunks is None


def test_empty_directory_writes_empty_chunk_list_to_store(tmp_path):
    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()
    store = FakeChromaStore()
    indexer = DocumentIndexer(store)

    count = indexer.index_directory(empty_dir, node_id="node-1")

    assert count == 0
    assert store.add_chunks_calls == 1
    assert store.added_chunks == []


def test_indexes_text_and_markdown_files_and_ignores_other_extensions(tmp_path):
    source_text = "A plain text document for indexing."
    source_markdown = "# Guide\n\nA Markdown document for indexing."
    write_text(tmp_path / "plain.TXT", source_text)
    write_text(tmp_path / "docs" / "guide.Md", source_markdown)
    write_text(tmp_path / "ignored.pdf", "This file must not be indexed.")
    write_text(tmp_path / "ignored.json", '{"text": "not indexed"}')
    store = FakeChromaStore()
    indexer = DocumentIndexer(store)

    count = indexer.index_directory(tmp_path, node_id="node-2")

    assert count == 2
    assert store.add_chunks_calls == 1
    assert {chunk.content for chunk in store.added_chunks} == {
        source_text,
        source_markdown,
    }
    assert {chunk.metadata["source"] for chunk in store.added_chunks} == {
        "plain.TXT",
        "guide.Md",
    }
    for chunk in store.added_chunks:
        assert chunk.node_id == "node-2"
        assert chunk.metadata == {
            "source": chunk.metadata["source"],
            "chunk_index": 0,
            "node_id": "node-2",
        }
        assert chunk.id.startswith(f"node-2_{Path(chunk.metadata['source']).stem}_0_")
        assert len(chunk.id.rsplit("_", maxsplit=1)[1]) == 6


def test_long_text_is_split_with_configured_size_and_overlap(tmp_path):
    content = "abcdefghijklmnopqrstuvwxyz0123456789"
    write_text(tmp_path / "long.txt", content)
    store = FakeChromaStore()
    indexer = DocumentIndexer(store, chunk_size=10, chunk_overlap=3)

    count = indexer.index_directory(tmp_path, node_id="node-3")
    chunks = store.added_chunks

    assert count == len(chunks)
    assert count > 1
    assert all(0 < len(chunk.content) <= 10 for chunk in chunks)
    assert all(chunk.content in content for chunk in chunks)
    assert [chunk.metadata["chunk_index"] for chunk in chunks] == list(
        range(count)
    )
    assert len({chunk.id for chunk in chunks}) == count
    assert all(chunk.node_id == "node-3" for chunk in chunks)


def test_empty_supported_file_produces_no_chunks(tmp_path):
    write_text(tmp_path / "empty.md", "")
    store = FakeChromaStore()
    indexer = DocumentIndexer(store)

    count = indexer.index_directory(tmp_path, node_id="node-4")

    assert count == 0
    assert store.add_chunks_calls == 1
    assert store.added_chunks == []


def test_jsonl_skips_invalid_and_blank_lines_and_preserves_record_metadata(
    tmp_path,
):
    jsonl = tmp_path / "posts.jsonl"
    records = [
        {
            "id": 42,
            "title_body": "Question body",
            "upvoted_answer": "Accepted answer",
            "domain": "legal",
        },
        {"id": 43, "title_body": "", "upvoted_answer": "", "domain": "legal"},
    ]
    jsonl.write_text(
        "\n".join(
            [
                json.dumps(records[0]),
                "not valid JSON",
                "",
                json.dumps(records[1]),
            ]
        ),
        encoding="utf-8",
    )
    store = FakeChromaStore()
    indexer = DocumentIndexer(store, chunk_size=100, chunk_overlap=0)

    count = indexer.index_directory(tmp_path, node_id="node-5")

    assert count == 1
    chunk = store.added_chunks[0]
    assert chunk.id == "node-5_posts_42_0"
    assert chunk.content == "Question body\n\nAccepted answer"
    assert chunk.node_id == "node-5"
    assert chunk.metadata == {
        "source": "posts.jsonl",
        "domain": "legal",
        "doc_id": 42,
        "chunk_index": 0,
        "node_id": "node-5",
    }


def test_jsonl_uses_line_number_and_default_values_when_fields_are_missing(
    tmp_path,
):
    write_text(
        tmp_path / "minimal.jsonl",
        "\n".join(
            [
                json.dumps({"title_body": "First post"}),
                "invalid line",
                "",
                json.dumps({"upvoted_answer": "Second answer"}),
            ]
        ),
    )
    store = FakeChromaStore()
    indexer = DocumentIndexer(store, chunk_size=100, chunk_overlap=0)

    count = indexer.index_directory(tmp_path, node_id="node-6")

    assert count == 2
    chunks_by_id = {chunk.id: chunk for chunk in store.added_chunks}
    first = chunks_by_id["node-6_minimal_0_0"]
    second = chunks_by_id["node-6_minimal_3_0"]
    assert first.content == "First post"
    assert first.metadata["domain"] == ""
    assert first.metadata["doc_id"] == 0
    assert second.content == "Second answer"
    assert second.metadata["domain"] == ""
    assert second.metadata["doc_id"] == 3


def test_jsonl_long_record_creates_ordered_chunks_with_stable_ids(tmp_path):
    content = "abcdefghijklmnopqrstuvwxyz"
    record = {
        "id": "question-1",
        "title_body": content,
        "upvoted_answer": "",
        "domain": "science",
    }
    write_text(tmp_path / "questions.jsonl", json.dumps(record))
    store = FakeChromaStore()
    indexer = DocumentIndexer(store, chunk_size=8, chunk_overlap=2)

    count = indexer.index_directory(tmp_path, node_id="node-7")

    chunks = store.added_chunks
    assert count == len(chunks)
    assert count > 1
    assert [chunk.id for chunk in chunks] == [
        f"node-7_questions_question-1_{index}" for index in range(count)
    ]
    assert [chunk.metadata["chunk_index"] for chunk in chunks] == list(
        range(count)
    )
    assert all(chunk.metadata["doc_id"] == "question-1" for chunk in chunks)
    assert all(chunk.metadata["domain"] == "science" for chunk in chunks)
    assert all(0 < len(chunk.content) <= 8 for chunk in chunks)
    assert all(chunk.content in content for chunk in chunks)


@pytest.mark.parametrize("directory_argument_kind", ["string", "path"])
def test_index_directory_accepts_string_and_path_arguments(
    tmp_path, directory_argument_kind
):
    write_text(tmp_path / "source.txt", "Text.")
    path_argument = (
        str(tmp_path) if directory_argument_kind == "string" else tmp_path
    )
    store = FakeChromaStore()
    indexer = DocumentIndexer(store)

    count = indexer.index_directory(path_argument, node_id="node-8")

    assert count == 1
    assert store.added_chunks[0].content == "Text."


def test_file_read_errors_are_not_silently_swallowed(tmp_path):
    bad_file = tmp_path / "invalid-utf8.txt"
    bad_file.write_bytes(b"\xff")
    store = FakeChromaStore()
    indexer = DocumentIndexer(store)

    with pytest.raises(UnicodeDecodeError):
        indexer.index_directory(tmp_path, node_id="node-9")

    assert store.add_chunks_calls == 0
