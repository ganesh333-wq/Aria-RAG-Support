"""
──────────────────────────────────────────────────────────────────────────────
 ChromaDB Vector Store Manager

 Handles:
   - Semantic chunking: 1 FAQ JSON object = 1 ChromaDB document
   - Chunk text = questions + tags + answer + category + intent combined
   - Persistent storage to disk (./chroma_db)
   - Cosine similarity search via ChromaDB's HNSW index
   - Metadata storage: id, category, intent, escalate, escalation_reason, tags

 Chunking Strategy:
   RULE: One FAQ object = One semantic chunk.
   NO RecursiveCharacterTextSplitter. NO fixed token windows. NO overlap.
   Each FAQ entry is already a complete semantic unit covering one topic.
──────────────────────────────────────────────────────────────────────────────
"""

import json
import os
import time
from typing import Any, Dict, List, Optional, Tuple

import chromadb
from chromadb.config import Settings
from langchain_chroma import Chroma
from langchain_core.documents import Document

from app.embeddings.minilm import get_embeddings


class ChromaStore:
    """
    ChromaDB-backed persistent vector store for Aria FAQ documents.

    Architecture:
        FAQ JSON → semantic chunk text → MiniLM embedding → ChromaDB HNSW index
        Query text → MiniLM embedding → cosine similarity → top-K documents
    """

    def __init__(
        self,
        persist_dir: str = "./chroma_db",
        collection_name: str = "aria_faq",
    ):
        self.persist_dir = persist_dir
        self.collection_name = collection_name
        self._vectorstore: Optional[Chroma] = None
        self._doc_count: int = 0
        self.stats: Dict[str, Any] = {
            "total_docs": 0,
            "embedding_dim": 384,
            "total_searches": 0,
            "avg_search_time_ms": 0.0,
            "is_indexed": False,
        }

    # ── Initialization ────────────────────────────────────────────────────────

    def initialize(self, kb_path: str) -> Dict[str, Any]:
        """
        Load knowledge base JSON, build semantic chunks, and index in ChromaDB.

        Semantic chunking strategy:
          For each FAQ entry, combine ALL fields into one rich text chunk:
            Question variants + Tags + Category + Intent + Answer
          This maximizes semantic coverage so retrieval catches paraphrases,
          synonyms, and related concepts.

        Args:
            kb_path: Path to knowledge_base.json

        Returns:
            dict with initialization statistics
        """
        start = time.time()

        # Load knowledge base
        with open(kb_path, "r") as f:
            knowledge_base: List[Dict] = json.load(f)

        print(f"[ChromaStore] Loaded {len(knowledge_base)} FAQ entries from {kb_path}")
        print(f"[ChromaStore] Building semantic chunks...")

        # Convert each FAQ entry into a LangChain Document
        documents: List[Document] = []
        ids: List[str] = []

        for entry in knowledge_base:
            chunk_text = self._build_semantic_chunk(entry)
            metadata = self._build_metadata(entry)

            documents.append(Document(
                page_content=chunk_text,
                metadata=metadata,
            ))
            ids.append(entry["id"])

        # Initialize embeddings
        embeddings = get_embeddings()

        # Create/overwrite ChromaDB collection
        # We always re-index on startup to stay in sync with knowledge_base.json
        os.makedirs(self.persist_dir, exist_ok=True)

        print(f"[ChromaStore] Indexing {len(documents)} chunks into ChromaDB...")
        print(f"[ChromaStore] Persist directory: {self.persist_dir}")

        # Build Chroma vectorstore with cosine similarity metric
        self._vectorstore = Chroma.from_documents(
            documents=documents,
            embedding=embeddings,
            ids=ids,
            collection_name=self.collection_name,
            persist_directory=self.persist_dir,
            collection_metadata={"hnsw:space": "cosine"},
        )

        self._doc_count = len(documents)
        elapsed = (time.time() - start) * 1000

        self.stats.update({
            "total_docs": self._doc_count,
            "embedding_dim": 384,
            "is_indexed": True,
            "initialization_time_ms": round(elapsed, 2),
        })

        print(
            f"[ChromaStore] Indexed {self._doc_count} chunks | "
            f"Time: {elapsed:.0f}ms | "
            f"Persist: {self.persist_dir}"
        )

        return self.stats

    # ── Semantic Chunking ─────────────────────────────────────────────────────

    def _build_semantic_chunk(self, entry: Dict) -> str:
        """
        Build a rich semantic chunk from a single FAQ entry.

        Combines all signal-bearing fields into one text string.
        The embedding model encodes this holistically, so a query about
        "send item back" will match the chunk containing "return policy"
        because of semantic similarity in the dense vector space.

        Approximate output: 150–400 tokens per chunk.
        """
        parts: List[str] = []

        # Questions and variants (these are the primary semantic anchors)
        for q in entry.get("questions", []):
            parts.append(f"Q: {q}")

        # Tags (keyword signals for boosting relevant dimensions)
        tags = entry.get("tags", [])
        if tags:
            parts.append(f"Keywords: {', '.join(tags)}")

        # Semantic Aliases (paraphrases for better embedding match)
        aliases = entry.get("semantic_aliases", [])
        if aliases:
            parts.append(f"Aliases: {', '.join(aliases)}")

        # Category and intent (topic anchors)
        if entry.get("category"):
            parts.append(f"Category: {entry['category']}")
        if entry.get("intent"):
            parts.append(f"Topic: {entry['intent'].replace('_', ' ').title()}")

        # Answer (the actual content — most weight for retrieval)
        answer = entry.get("answer")
        if answer:
            parts.append(f"Answer: {answer}")
        elif entry.get("escalate"):
            # Escalation docs: describe the scenario for semantic matching
            reason = entry.get("escalation_reason", "")
            parts.append(f"Situation: {reason} This issue requires human support.")

        return "\n".join(parts)

    def _build_metadata(self, entry: Dict) -> Dict:
        """
        Build ChromaDB metadata dict for a FAQ entry.
        ChromaDB metadata values must be: str, int, float, or bool.
        """
        return {
            "id": entry["id"],
            "category": entry.get("category", ""),
            "intent": entry.get("intent", ""),
            "escalate": bool(entry.get("escalate", False)),
            "escalation_reason": entry.get("escalation_reason") or "",
            "tags": ", ".join(entry.get("tags", [])),
            "semantic_aliases": ", ".join(entry.get("semantic_aliases", [])),
            "priority": entry.get("priority", "medium"),
        }

    # ── Search ────────────────────────────────────────────────────────────────

    def similarity_search(
        self,
        query: str,
        k: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        Perform cosine similarity search.

        Converts query to MiniLM embedding, searches ChromaDB HNSW index,
        and returns top-K results with relevance scores in [0, 1].

        Args:
            query: Raw user query (will be embedded)
            k: Number of top documents to return

        Returns:
            List of result dicts with: id, score, category, intent, text,
                                       escalate, escalation_reason, tags
        """
        if not self._vectorstore:
            raise RuntimeError("ChromaStore not initialized. Call initialize() first.")

        start = time.time()

        # similarity_search_with_relevance_scores returns List[Tuple[Document, float]]
        # Score is in [0, 1] where 1 = most similar (for cosine space)
        raw_results: List[Tuple[Document, float]] = (
            self._vectorstore.similarity_search_with_relevance_scores(query, k=k)
        )

        results = []
        for doc, score in raw_results:
            meta = doc.metadata
            results.append({
                "id": meta.get("id", ""),
                "score": round(float(score), 4),
                "category": meta.get("category", ""),
                "intent": meta.get("intent", ""),
                "text": doc.page_content,
                "escalate": bool(meta.get("escalate", False)),
                "escalation_reason": meta.get("escalation_reason", "") or None,
                "tags": [t.strip() for t in meta.get("tags", "").split(",") if t.strip()],
            })

        elapsed_ms = (time.time() - start) * 1000
        self.stats["total_searches"] += 1
        self.stats["avg_search_time_ms"] = round(
            (self.stats["avg_search_time_ms"] * (self.stats["total_searches"] - 1) + elapsed_ms)
            / self.stats["total_searches"],
            2,
        )

        return results

    def get_langchain_retriever(self, k: int = 5):
        """
        Return a LangChain VectorStoreRetriever for use in LangChain chains.
        Compatible with ConversationalRetrievalChain.
        """
        if not self._vectorstore:
            raise RuntimeError("ChromaStore not initialized. Call initialize() first.")
        return self._vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": k},
        )

    def get_stats(self) -> Dict[str, Any]:
        return {**self.stats}
