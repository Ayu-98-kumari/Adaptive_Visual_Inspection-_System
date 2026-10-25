"""
Retriever Module for Hybrid Semantic & Metadata Search
"""

from typing import List, Dict, Any, Optional
from rag.embeddings import EmbeddingModel
from rag.vector_store import VectorStore
from rag.config import RAGConfig

class RAGRetriever:
    def __init__(self, vector_store: VectorStore, embedding_model: EmbeddingModel):
        self.vector_store = vector_store
        self.embedding_model = embedding_model

    def retrieve(
        self,
        query: str,
        category: Optional[str] = None,
        defect_code: Optional[str] = None,
        top_k: int = RAGConfig.SIMILARITY_TOP_K
    ) -> List[Dict[str, Any]]:
        """Encodes query and retrieves top matching chunks from the vector store."""
        query_vec = self.embedding_model.encode_one(query)
        return self.vector_store.query(
            query_vec=query_vec,
            top_k=top_k,
            category=category,
            defect_code=defect_code
        )

    def format_context_for_prompt(self, chunks: List[Dict[str, Any]]) -> str:
        """Formats retrieved chunks into an authoritative prompt context block with citations."""
        if not chunks:
            return "No specific SOP documentation matched this query."

        context_blocks = []
        for i, c in enumerate(chunks, 1):
            block = (
                f"[Source Ref {i}: {c.get('doc_id', 'SOP')} - {c.get('section_title', 'General')}]\n"
                f"Category: {c.get('category', 'all').upper()} | Defect Code: {c.get('defect_code', 'general')}\n"
                f"{c.get('content', '').strip()}\n"
            )
            context_blocks.append(block)

        return "\n----------------------------------------\n".join(context_blocks)
