import os
import numpy as np
from typing import List, Tuple, Any
from sentence_transformers import CrossEncoder
from config import config


class Reranker:
    """Advanced reranking using cross-encoder and diversity filtering"""

    def __init__(self):
        self.method = config.RERANK_METHOD
        self.model = None
        if self.method == "cross_encoder":
            try:
                self.model = CrossEncoder(config.RERANK_MODEL)
            except Exception as e:
                print(f"Warning: Could not load cross-encoder model: {e}")
                self.method = "basic"

    def rerank(self, query: str, documents: List[Any], scores: List[float], top_k: int = 3) -> List[Tuple[Any, float]]:
        """Rerank documents based on query relevance"""
        if self.method == "cross_encoder" and self.model:
            return self._cross_encoder_rerank(query, documents, scores, top_k)
        elif self.method == "mmr":
            return self._mmr_rerank(query, documents, scores, top_k)
        else:
            # Basic: just sort by original scores
            sorted_indices = np.argsort(scores)[::-1][:top_k]
            return [(documents[i], scores[i]) for i in sorted_indices]

    def _cross_encoder_rerank(self, query: str, documents: List[Any], scores: List[float], top_k: int) -> List[Tuple[Any, float]]:
        """Rerank using cross-encoder model"""
        doc_texts = [doc.page_content for doc in documents]

        # Batch processing for performance
        cross_scores = self._batch_predict(query, doc_texts)

        # Combine vector similarity with cross-encoder scores
        vector_weight = 1 - config.DIVERSITY_WEIGHT
        combined_scores = vector_weight * np.array(scores) + config.DIVERSITY_WEIGHT * cross_scores

        # Get top-k indices
        top_indices = np.argsort(combined_scores)[::-1][:top_k]

        return [(documents[i], combined_scores[i]) for i in top_indices]

    def _mmr_rerank(self, query: str, documents: List[Any], scores: List[float], top_k: int) -> List[Tuple[Any, float]]:
        """Maximal Marginal Relevance reranking for diversity"""
        selected = []
        remaining = list(zip(documents, scores))

        for _ in range(min(top_k, len(remaining))):
            if not remaining:
                break

            best_score = -float('inf')
            best_idx = 0

            for i, (doc, score) in enumerate(remaining):
                # Relevance score
                relevance = score

                # Diversity penalty (simplified)
                diversity_penalty = 0
                if selected:
                    # Simple diversity based on source
                    current_source = doc.metadata.get("file", "")
                    selected_sources = [s[0].metadata.get("file", "") for s in selected]
                    if current_source in selected_sources:
                        diversity_penalty = 0.1  # Penalty for same source

                mmr_score = config.DIVERSITY_WEIGHT * relevance - (1 - config.DIVERSITY_WEIGHT) * diversity_penalty

                if mmr_score > best_score:
                    best_score = mmr_score
                    best_idx = i

            selected.append(remaining[best_idx])
            remaining.pop(best_idx)

        return selected

    def _batch_predict(self, query: str, documents: List[str]) -> np.ndarray:
        """Batch predict relevance scores"""
        pairs = [(query, doc) for doc in documents]
        batch_size = config.RERANK_BATCH_SIZE

        scores = []
        for i in range(0, len(pairs), batch_size):
            batch = pairs[i:i + batch_size]
            batch_scores = self.model.predict(batch)
            scores.extend(batch_scores)

        return np.array(scores)


# Global reranker instance
reranker = Reranker()


def get_top_relevant_sources(results: List[Tuple[Any, float]], k: int = 3) -> List[str]:
    """Extract and format source information from reranked results"""
    sources = []
    for doc, score in results[:k]:
        metadata = doc.metadata
        file = metadata.get("file", "unknown")
        page = metadata.get("page", "N/A")
        chunk_id = metadata.get("count", "N/A")
        filename = os.path.basename(file)

        # More informative source citation with score
        source_info = f"{filename} (page {page}) - Relevance: {score:.3f}"
        sources.append(source_info)

    # Remove duplicates while preserving order
    seen = set()
    unique_sources = []
    for source in sources:
        if source not in seen:
            unique_sources.append(source)
            seen.add(source)

    return unique_sources


def get_context_from_documents(results: List[Tuple[Any, float]], k: int = 3) -> Tuple[str, List[str]]:
    """Extract context and sources from search results with basic reranking (no query)"""
    # Apply reranking without query (fallback)
    reranked_results = reranker.rerank("", [doc for doc, _ in results], [score for _, score in results], k)

    # Extract sources
    sources = get_top_relevant_sources(reranked_results, k)

    # Join document content
    enhanced_context_text = "\n\n---\n\n".join([doc.page_content for doc, _ in reranked_results])

    return enhanced_context_text, sources


def get_context_from_documents_with_query(query: str, results: List[Tuple[Any, float]], k: int = 3) -> Tuple[str, List[str]]:
    """Enhanced version that uses query for better reranking"""
    # Extract documents and scores
    documents = [doc for doc, _ in results]
    scores = [score for _, score in results]

    # Apply reranking with query
    reranked_results = reranker.rerank(query, documents, scores, k)

    # Extract sources
    sources = get_top_relevant_sources(reranked_results, k)

    # Join document content
    enhanced_context_text = "\n\n---\n\n".join([doc.page_content for doc, _ in reranked_results])

    return enhanced_context_text, sources
