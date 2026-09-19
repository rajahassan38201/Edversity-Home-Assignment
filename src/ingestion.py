"""
Ingestion pipeline: parse markdown files → ChromaDB with Mistral embeddings.
"""
import os
import re
from pathlib import Path
from typing import List

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_mistralai import MistralAIEmbeddings

load_dotenv()


def parse_markdown_file(file_path: Path, source_type: str) -> List[Document]:
    """Parse a markdown file into chunks with metadata."""
    content = file_path.read_text(encoding="utf-8")
    documents: List[Document] = []

    # Split by newline followed by "# FAQ-XX", "# POLICY-XX", or "# TICKET-XX"
    sections = re.split(r"\n(?=# (?:FAQ|POLICY|TICKET)-\d+)", content)
    
    for section in sections:
        section = section.strip()
        if not section:
            continue

        id_match = re.search(r"#\s*((?:FAQ|POLICY|TICKET)-\d+)", section)
        if not id_match:
            continue
            
        doc_id = id_match.group(1)

        title_match = re.search(r"—\s*(.+?)(?:\n|$)", section)
        topic = title_match.group(1).strip().lower() if title_match else "general"

        stale_markers = [
            "older version", "outdated", "obsolete", "no longer",
            "previous version", "retired", "archived", "has been removed",
        ]
        is_outdated = any(marker in section.lower() for marker in stale_markers)

        status = None
        if source_type == "ticket":
            status_match = re.search(r"STATUS:\s*(.+?)(?:\n|$)", section)
            status = status_match.group(1).strip() if status_match else "unknown"

        last_reviewed = None
        if source_type == "policy":
            date_match = re.search(
                r"(?:Last reviewed|Effective date|Updated|Reviewed|Last updated):\s*(.+?)(?:\n|$)",
                section, re.IGNORECASE,
            )
            last_reviewed = date_match.group(1).strip() if date_match else None

        doc = Document(
            page_content=section,
            metadata={
                "source_type": source_type,
                "source_id": doc_id,
                "topic": topic,
                "is_outdated": is_outdated,
                "status": status,
                "last_reviewed": last_reviewed,
            },
        )
        documents.append(doc)

    return documents


def ingest_data(data_dir: str = "data", persist_dir: str = "./chroma_db"):
    """Ingest all markdown files into a single ChromaDB collection."""
    mistral_key = os.getenv("MISTRAL_API_KEY")
    if not mistral_key:
        raise RuntimeError("MISTRAL_API_KEY not set. Copy .env.example → .env and fill it in.")

    # Use Mistral's state-of-the-art embedding model
    embeddings = MistralAIEmbeddings(
        model="mistral-embed",
        mistral_api_key=mistral_key,
    )

    # Wipe and recreate to avoid duplicates on re-run
    try:
        old = Chroma(
            collection_name="learnforge_knowledge",
            embedding_function=embeddings,
            persist_directory=persist_dir,
        )
        old.delete_collection()
    except Exception:
        pass

    vectorstore = Chroma(
        collection_name="learnforge_knowledge",
        embedding_function=embeddings,
        persist_directory=persist_dir,
    )

    files_to_process = [
        ("faqs.md", "faq"),
        ("policies.md", "policy"),
        ("tickets.md", "ticket"),
    ]

    all_docs: List[Document] = []
    for filename, source_type in files_to_process:
        file_path = Path(data_dir) / filename
        if file_path.exists():
            docs = parse_markdown_file(file_path, source_type)
            all_docs.extend(docs)
            print(f"✓ Parsed {len(docs)} sections from {filename}")
        else:
            print(f"✗ File not found: {file_path}")

    if all_docs:
        vectorstore.add_documents(all_docs)
        print(f"\n✅ SUCCESS: Added {len(all_docs)} documents to vector store")
        print(f"✓ Embeddings model: mistral-embed")
        print(f"✓ Persisted at: {persist_dir}")
    else:
        print("\n⚠ No documents found.")

    return vectorstore


if __name__ == "__main__":
    ingest_data()