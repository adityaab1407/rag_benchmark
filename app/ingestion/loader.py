import fitz
import re
import json
from pathlib import Path
from typing import List
from dataclasses import dataclass
from loguru import logger


@dataclass
class Document:
    content: str
    metadata: dict


def load_pdf(filepath: str) -> Document:
    doc = fitz.open(filepath)
    text = ""
    for page in doc:
        text += page.get_text()

    if len(text) < 100:
        logger.warning("{} - very little text, may be scanned PDF", Path(filepath).name)

    return Document(
        content=text,
        metadata={
            "source": filepath,
            "filename": Path(filepath).name,
            "doc_type": "pdf",
            "page_count": len(doc),
            "char_count": len(text)
        }
    )


def load_markdown(filepath: str) -> Document:
    path = Path(filepath)
    content = path.read_text(encoding="utf-8", errors="ignore")

    return Document(
        content=content,
        metadata={
            "source": filepath,
            "filename": path.name,
            "doc_type": "markdown",
            "char_count": len(content)
        }
    )


def load_text(filepath: str) -> Document:
    path = Path(filepath)
    content = path.read_text(encoding="utf-8", errors="ignore")

    return Document(
        content=content,
        metadata={
            "source": filepath,
            "filename": path.name,
            "doc_type": "text",
            "char_count": len(content)
        }
    )


def load_html(filepath: str) -> Document:
    path = Path(filepath)
    raw = path.read_text(encoding="utf-8", errors="ignore")

    raw = re.sub(r'<script[^>]*>.*?</script>', '', raw, flags=re.DOTALL)
    raw = re.sub(r'<style[^>]*>.*?</style>', '', raw, flags=re.DOTALL)
    text = re.sub(r'<[^>]+>', ' ', raw)
    text = re.sub(r'\s+', ' ', text).strip()

    return Document(
        content=text,
        metadata={
            "source": filepath,
            "filename": path.name,
            "doc_type": "html",
            "char_count": len(text)
        }
    )


def load_jsonl(filepath: str) -> Document:
    texts = []

    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                text = obj.get("text", "") or obj.get("title", "")
                if text:
                    texts.append(text)
            except json.JSONDecodeError:
                continue

    content = "\n\n".join(texts)

    return Document(
        content=content,
        metadata={
            "source": filepath,
            "filename": Path(filepath).name,
            "doc_type": "jsonl",
            "char_count": len(content),
            "document_count": len(texts)
        }
    )


def load_all_documents(directory: str) -> List[Document]:
    docs = []
    path = Path(directory)

    loaders = {
        ".pdf":   load_pdf,
        ".md":    load_markdown,
        ".txt":   load_text,
        ".html":  load_html,
        ".jsonl": load_jsonl,
    }

    for file in path.rglob("*"):
        if not file.is_file():
            continue

        suffix = file.suffix.lower()
        if suffix not in loaders:
            continue

        try:
            doc = loaders[suffix](str(file))

            if len(doc.content.strip()) < 100:
                logger.debug("Skipping {} - too short", file.name)
                continue

            docs.append(doc)
            logger.info("Loaded: {} - {:,} chars", file.name, doc.metadata["char_count"])

        except Exception as e:
            logger.warning("Failed to load {}: {}", file.name, e)
            continue

    logger.info("Total loaded: {} documents", len(docs))
    return docs