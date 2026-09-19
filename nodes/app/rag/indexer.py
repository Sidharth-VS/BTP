from pathlib import Path
from typing import List
import uuid
from langchain_text_splitters import RecursiveCharacterTextSplitter
from nodes.app.rag.chroma_db import ChromaStore
from shared.schemas.document import DocumentChunk


class DocumentIndexer:
    def __init__(
        self,
        chroma_store: ChromaStore,
        chunk_size: int = 500,
        chunk_overlap: int = 50,
    ):
        self.chroma_store = chroma_store
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

    def index_directory(self, dir_path: str | Path, node_id: str) -> int:
        """Reads all txt/md files in directory, chunks them, and stores in ChromaDB."""
        path = Path(dir_path)
        if not path.exists():
            return 0

        chunks: List[DocumentChunk] = []
        for file_path in path.glob("**/*.*"):
            if file_path.suffix.lower() not in [".txt", ".md"]:
                continue

            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()

            split_texts = self.text_splitter.split_text(content)
            for idx, text in enumerate(split_texts):
                chunk_id = f"{node_id}_{file_path.stem}_{idx}_{uuid.uuid4().hex[:6]}"
                chunks.append(
                    DocumentChunk(
                        id=chunk_id,
                        content=text,
                        metadata={
                            "source": str(file_path.name),
                            "chunk_index": idx,
                            "node_id": node_id,
                        },
                        node_id=node_id,
                    )
                )

        self.chroma_store.add_chunks(chunks)
        return len(chunks)