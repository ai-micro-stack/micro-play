import os
import chromadb
from chromadb.errors import NotFoundError
from config import config


class ChromaEmbeddingFunction(chromadb.EmbeddingFunction):
    def __init__(self, embedding_function):
        self.embedding_function = embedding_function

    def __call__(self, input):
        """ChromaDB EmbeddingFunction interface: input is list of texts, return list of embeddings."""
        return self.embedding_function.embed_documents(input)


class PlatServedChromaDb:
    def __init__(self, vectordb_provider: str, api_url: str, api_key: str = None):
        self.vectordb_provider = vectordb_provider
        self.api_url = api_url
        self.api_key = api_key

        # Create persistent directory path
        self.vectordb_path = os.path.join(
            config.VECTORDB_ROOT, f"chroma-{vectordb_provider}"
        )
        os.makedirs(self.vectordb_path, exist_ok=True)

        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(path=self.vectordb_path)
        self.collection_name = "rag_documents"
        self.embedding_function = None
        self.collection = None

    def set_embedding_function(self, embedding_function):
        """Set the embedding function and initialize/create collection."""
        self.embedding_function = ChromaEmbeddingFunction(embedding_function)

        # Create or get existing collection
        try:
            self.collection = self.client.get_collection(
                name=self.collection_name,
                embedding_function=self.embedding_function
            )
        except NotFoundError:
            # Collection doesn't exist, create it
            self.collection = self.client.create_collection(
                name=self.collection_name,
                embedding_function=self.embedding_function
            )

    def store_the_chunks(self, chunks):
        """Store document chunks in the vector database."""
        if not self.collection:
            raise ValueError("Collection not initialized. Call set_embedding_function first.")

        documents = []
        metadatas = []
        ids = []

        for chunk in chunks:
            documents.append(chunk["text"])
            metadatas.append({
                "source": chunk["id"],
                "file": chunk["file"],
                "page": chunk["page"],
                "line": chunk["line"],
                "count": chunk["count"]
            })
            ids.append(chunk["id"])

        self.collection.add(
            documents=documents,
            metadatas=metadatas,
            ids=ids
        )

    def persist_vector_store(self):
        """Persist the vector store (ChromaDB handles this automatically)."""
        pass

    def search_similar_chunks(self, query_text, k=5):
        """Search for similar chunks and return results with scores."""
        if not self.collection:
            raise ValueError("Collection not initialized. Call set_embedding_function first.")

        results = self.collection.query(
            query_texts=[query_text],
            n_results=k,
            include=['documents', 'metadatas', 'distances']
        )

        # Format results to match the expected interface
        formatted_results = []
        if results['documents'] and results['metadatas'] and results['distances']:
            for doc, metadata, distance in zip(
                results['documents'][0],
                results['metadatas'][0],
                results['distances'][0]
            ):
                # Create a mock document object similar to LangChain's format
                mock_doc = MockDocument(
                    page_content=doc,
                    metadata=metadata
                )
                formatted_results.append((mock_doc, distance))

        return formatted_results

    def check_file_is_indexed(self, file_name):
        """Check if a file has been indexed."""
        if not self.collection:
            return False

        try:
            # Query for documents with this file
            results = self.collection.query(
                query_texts=["dummy"],
                n_results=1,
                where={"file": file_name},
                include=[]
            )
            return len(results['ids'][0]) > 0 if results['ids'] else False
        except Exception:
            return False


class MockDocument:
    """Mock document class to maintain compatibility with existing code."""
    def __init__(self, page_content, metadata):
        self.page_content = page_content
        self.metadata = metadata
