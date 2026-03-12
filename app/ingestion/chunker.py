from typing import List
from dataclasses import dataclass
from loguru import logger
from app.ingestion.loader import Document


@dataclass
class Chunk:
    content: str
    metadata: dict
    chunk_index: int
    strategy: str


def fixed_size_chunk(doc, chunk_size=512, overlap=50):
    text = doc.content
    chunks = []
    start = 0
    index = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunk_text = text[start:end]
        if chunk_text.strip():
            chunks.append(Chunk(
                content=chunk_text,
                metadata={**doc.metadata, "chunk_start": start, "chunk_end": end},
                chunk_index=index,
                strategy="fixed_size"
            ))
            index += 1
        start += chunk_size - overlap
    logger.debug("fixed_size: {} chunks from {}", len(chunks), doc.metadata["filename"])
    return chunks


def semantic_chunk(doc, max_chunk_size=600):
    text = doc.content
    chunks = []
    index = 0
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    current_chunk = ""
    for para in paragraphs:
        if len(current_chunk) + len(para) + 2 <= max_chunk_size:
            current_chunk += ("\n\n" if current_chunk else "") + para
        else:
            if current_chunk.strip():
                chunks.append(Chunk(
                    content=current_chunk.strip(),
                    metadata=doc.metadata,
                    chunk_index=index,
                    strategy="semantic"
                ))
                index += 1
            if len(para) > max_chunk_size:
                for i in range(0, len(para), max_chunk_size - 50):
                    sub = para[i:i + max_chunk_size]
                    if sub.strip():
                        chunks.append(Chunk(
                            content=sub.strip(),
                            metadata=doc.metadata,
                            chunk_index=index,
                            strategy="semantic"
                        ))
                        index += 1
                current_chunk = ""
            else:
                current_chunk = para
    if current_chunk.strip():
        chunks.append(Chunk(
            content=current_chunk.strip(),
            metadata=doc.metadata,
            chunk_index=index,
            strategy="semantic"
        ))
    logger.debug("semantic: {} chunks from {}", len(chunks), doc.metadata["filename"])
    return chunks


def parent_child_chunk(doc, parent_size=1000, child_size=256):
    text = doc.content
    chunks = []
    parent_index = 0
    parent_start = 0
    while parent_start < len(text):
        parent_end = min(parent_start + parent_size, len(text))
        parent_text = text[parent_start:parent_end]
        parent_id = "{}_parent_{}".format(doc.metadata["filename"], parent_index)
        if parent_text.strip():
            chunks.append(Chunk(
                content=parent_text,
                metadata={**doc.metadata, "chunk_type": "parent", "parent_id": parent_id},
                chunk_index=parent_index,
                strategy="parent_child"
            ))
        child_start = parent_start
        child_index = 0
        while child_start < parent_end:
            child_end = min(child_start + child_size, parent_end)
            child_text = text[child_start:child_end]
            if child_text.strip():
                chunks.append(Chunk(
                    content=child_text,
                    metadata={**doc.metadata, "chunk_type": "child", "parent_id": parent_id},
                    chunk_index=child_index,
                    strategy="parent_child"
                ))
                child_index += 1
            child_start += child_size
        parent_start += parent_size
        parent_index += 1
    logger.debug("parent_child: {} chunks from {}", len(chunks), doc.metadata["filename"])
    return chunks


def chunk_document_all_strategies(doc):
    all_chunks = []
    all_chunks.extend(fixed_size_chunk(doc))
    all_chunks.extend(semantic_chunk(doc))
    all_chunks.extend(parent_child_chunk(doc))
    return all_chunks
