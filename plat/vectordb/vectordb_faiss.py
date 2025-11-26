import os
import faiss
import pickle
import numpy as np
"""
FAISS vector database implementation for the RAG system.

This module provides a FAISS-based vector store that supports storing document chunks,
searching for similar content, and persisting the index to disk.
"""

from config import config


class PlatServedFaissDb:
    """
    FAISS-based vector database implementation.

    Provides methods for storing document embeddings, searching for similar chunks,
    and persisting the index and metadata to disk.
    """
    def __init__(self, vectordb_provider: str, api_url: str, api_key: str = None):
        self.vectordb_provider = vectordb_provider
        self.api_url = api_url
        self.api_key = api_key

        # Setup paths
        self.db_dir = os.path.join(config.VECTORDB_ROOT, f"faiss-{vectordb_provider}")
        os.makedirs(self.db_dir, exist_ok=True)

        self.index_path = os.path.join(self.db_dir, "index.faiss")
        self.metadata_path = os.path.join(self.db_dir, "metadata.pkl")
        self.id_map_path = os.path.join(self.db_dir, "id_map.pkl")

        # Initialize components
        self.embedding_function = None
        self.index = None
        self.metadata = []  # List of chunk metadata
        self.id_to_idx = {}  # Maps chunk IDs to FAISS indices

        # Load existing index if available
        self._load_index()

    def set_embedding_function(self, embedding_function):
        """Set the embedding function and initialize FAISS index."""
        self.embedding_function = embedding_function

        if self.index is None:
            # Get embedding dimension
            test_embedding = embedding_function.embed_query("test")
            embedding_dim = len(test_embedding)

            # Create FAISS index
            self.index = faiss.IndexFlatL2(embedding_dim)

    def _load_index(self):
        """Load existing FAISS index and metadata if available."""
        try:
            if os.path.exists(self.index_path):
                self.index = faiss.read_index(self.index_path)

            if os.path.exists(self.metadata_path):
                with open(self.metadata_path, 'rb') as f:
                    self.metadata = pickle.load(f)

            if os.path.exists(self.id_map_path):
                with open(self.id_map_path, 'rb') as f:
                    self.id_to_idx = pickle.load(f)

        except Exception as e:
            print(f"Warning: Could not load existing FAISS index: {e}")
            self.index = None
            self.metadata = []
            self.id_to_idx = {}

    def store_the_chunks(self, chunks):
        """Store document chunks in the FAISS index."""
        if not self.embedding_function or not self.index:
            raise ValueError("Embedding function not set. Call set_embedding_function first.")

        # Prepare data for batch insertion
        texts = [chunk["text"] for chunk in chunks]
        embeddings = np.array(self.embedding_function.embed_documents(texts))

        # Add to FAISS index
        start_idx = len(self.metadata)
        self.index.add(embeddings.astype('float32'))

        # Store metadata
        for i, chunk in enumerate(chunks):
            chunk_metadata = {
                "id": chunk["id"],
                "file": chunk["file"],
                "page": chunk["page"],
                "line": chunk["line"],
                "count": chunk["count"],
                "text": chunk["text"],
                "faiss_idx": start_idx + i
            }
            self.metadata.append(chunk_metadata)
            self.id_to_idx[chunk["id"]] = start_idx + i

    def persist_vector_store(self):
        """Persist the FAISS index and metadata to disk."""
        if self.index:
            faiss.write_index(self.index, self.index_path)

        with open(self.metadata_path, 'wb') as f:
            pickle.dump(self.metadata, f)

        with open(self.id_map_path, 'wb') as f:
            pickle.dump(self.id_to_idx, f)

    def search_similar_chunks(self, query_text, k=5):
        """Search for similar chunks using FAISS."""
        if not self.embedding_function or not self.index or self.index.ntotal == 0:
            return []

        # Embed query
        query_embedding = np.array(self.embedding_function.embed_query(query_text)).astype('float32').reshape(1, -1)

        # Search FAISS index
        distances, indices = self.index.search(query_embedding, min(k, self.index.ntotal))

        # Format results
        results = []
        for distance, idx in zip(distances[0], indices[0]):
            if idx < len(self.metadata):
                chunk_data = self.metadata[idx]
                # Create mock document for compatibility
                mock_doc = MockDocument(
                    page_content=chunk_data["text"],
                    metadata={
                        "source": chunk_data["id"],
                        "file": chunk_data["file"],
                        "page": chunk_data["page"],
                        "line": chunk_data["line"],
                        "count": chunk_data["count"]
                    }
                )
                results.append((mock_doc, float(distance)))

        return results

    def check_file_is_indexed(self, file_name):
        """Check if a file has been indexed."""
        return any(chunk["file"] == file_name for chunk in self.metadata)

    def convert_index_to_tsv(self, full_data=False):
        """Convert FAISS index to TSV format for visualization."""
        if not self.index:
            return {"error": "No index available"}

        try:
            # Reconstruct vectors
            vectors = self.index.reconstruct_n(0, self.index.ntotal)
            tsv_path = os.path.join(self.db_dir, "vectors.tsv")
            np.savetxt(tsv_path, vectors, delimiter="\t")

            return {
                "projector-url": "https://projector.tensorflow.org/",
                "vectors-tsv": tsv_path,
            }
        except Exception as e:
            return {"error": f"Failed to convert index: {e}"}


class MockDocument:
    """Mock document class to maintain compatibility with existing code."""
    def __init__(self, page_content, metadata):
        self.page_content = page_content
        self.metadata = metadata
