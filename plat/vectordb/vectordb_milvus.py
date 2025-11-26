import os
from pymilvus import connections, Collection, CollectionSchema, FieldSchema, DataType
from config import config


class PlatServedMilvusDb:
    def __init__(self, vectordb_provider: str, api_url: str, api_key: str = None):
        self.vectordb_provider = vectordb_provider
        self.api_url = api_url
        self.api_key = api_key

        # Setup collection name and connection
        self.collection_name = f"rag_documents_{vectordb_provider}"
        self.embedding_function = None
        self.collection = None

        # Connect to Milvus
        connect_kwargs = {
            "alias": "default"
        }
        if self.api_url:
            connect_kwargs["uri"] = self.api_url
        if self.api_key:
            connect_kwargs["token"] = self.api_key
        connections.connect(**connect_kwargs)

    def set_embedding_function(self, embedding_function):
        """Set the embedding function and initialize/create collection."""
        self.embedding_function = embedding_function

        try:
            # Try to get existing collection
            self.collection = Collection(self.collection_name)
        except Exception:
            # Collection doesn't exist, create it
            self._create_collection()

    def _create_collection(self):
        """Create the Milvus collection with proper schema."""
        if not self.embedding_function:
            raise ValueError("Embedding function not set")

        # Get embedding dimension
        test_embedding = self.embedding_function.embed_query("test")
        embedding_dim = len(test_embedding)

        # Define schema
        fields = [
            FieldSchema(name="id", dtype=DataType.VARCHAR, max_length=512, is_primary=True),
            FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=embedding_dim),
            FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535),
            FieldSchema(name="file", dtype=DataType.VARCHAR, max_length=512),
            FieldSchema(name="page", dtype=DataType.INT64),
            FieldSchema(name="line", dtype=DataType.VARCHAR, max_length=512),
            FieldSchema(name="count", dtype=DataType.INT64),
        ]

        schema = CollectionSchema(fields, description="RAG document chunks")
        self.collection = Collection(self.collection_name, schema)

        # Create index on vector field
        index_params = {
            "metric_type": "L2",
            "index_type": "FLAT",
        }
        self.collection.create_index("vector", index_params)
        self.collection.load()

    def store_the_chunks(self, chunks):
        """Store document chunks in Milvus."""
        if not self.collection or not self.embedding_function:
            raise ValueError("Collection not initialized. Call set_embedding_function first.")

        # Prepare data for batch insertion
        ids = []
        vectors = []
        texts = []
        files = []
        pages = []
        lines = []
        counts = []

        for chunk in chunks:
            # Generate embeddings
            vector = self.embedding_function.embed_query(chunk["text"])

            ids.append(chunk["id"])
            vectors.append(vector)
            texts.append(chunk["text"])
            files.append(chunk["file"])
            pages.append(chunk["page"])
            lines.append(chunk["line"])
            counts.append(chunk["count"])

        # Insert data
        entities = [ids, vectors, texts, files, pages, lines, counts]
        self.collection.insert(entities)
        self.collection.flush()

    def search_similar_chunks(self, query_text, k=5):
        """Search for similar chunks in Milvus."""
        if not self.collection or not self.embedding_function:
            return []

        # Generate query embedding
        query_vector = self.embedding_function.embed_query(query_text)

        # Search parameters
        search_params = {
            "metric_type": "L2",
            "params": {}
        }

        # Perform search
        results = self.collection.search(
            data=[query_vector],
            anns_field="vector",
            param=search_params,
            limit=k,
            output_fields=["text", "file", "page", "line", "count"]
        )

        # Format results to match expected interface
        formatted_results = []
        if results and results[0]:
            for hit in results[0]:
                mock_doc = MockDocument(
                    page_content=hit.entity.get('text', ''),
                    metadata={
                        "source": hit.id,
                        "file": hit.entity.get('file', ''),
                        "page": hit.entity.get('page', 0),
                        "line": hit.entity.get('line', ''),
                        "count": hit.entity.get('count', 0)
                    }
                )
                formatted_results.append((mock_doc, float(hit.distance)))

        return formatted_results

    def check_file_is_indexed(self, file_name):
        """Check if a file has been indexed."""
        if not self.collection:
            return False

        try:
            # Query for documents with this file
            # Escape backslashes for Milvus expression parser
            escaped_file_name = file_name.replace('\\', '\\\\')
            expr = f'file == "{escaped_file_name}"'
            results = self.collection.query(
                expr=expr,
                limit=1,
                output_fields=[]
            )
            return len(results) > 0
        except Exception:
            return False

    def persist_vector_store(self):
        """Persist the vector store (Milvus handles this automatically)."""
        if self.collection:
            self.collection.flush()


class MockDocument:
    """Mock document class to maintain compatibility with existing code."""
    def __init__(self, page_content, metadata):
        self.page_content = page_content
        self.metadata = metadata
