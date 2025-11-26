import os
import json
import requests
import numpy as np
from dotenv import load_dotenv

load_dotenv()
EMBEDDING_API_URL = os.getenv("EMBEDDING_API_URL")


class PlatServedEmbeddings:
    """
    Embedding provider that interfaces with a served embedding model API.

    This class implements methods compatible with ChromaDB's embedding function interface,
    allowing batch embedding of documents and single query embedding.
    """

    def __init__(self, model: str):
        """
        Initialize the embedding provider.

        Args:
            model (str): The name of the embedding model to use.
        """
        self.model = model
        self.embedding_api_url = EMBEDDING_API_URL
        self.api_endpoint = self.embedding_api_url + "/api/embed"

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        """
        Embed a list of documents.

        Args:
            texts (list[str]): List of text documents to embed.

        Returns:
            np.ndarray: Array of embedding vectors.
        """
        payload = {"model": self.model, "input": texts}
        response = requests.post(self.api_endpoint, json=payload)
        vectors = response.json()["embeddings"]
        return np.array(vectors)

    def embed_query(self, text: str) -> np.ndarray:
        """
        Embed a single query text.

        Args:
            text (str): The query text to embed.

        Returns:
            np.ndarray: The embedding vector for the query.
        """
        payload = {"model": self.model, "input": [text]}
        response = requests.post(self.api_endpoint, json=payload)
        vectors = response.json()["embeddings"]
        return np.array(vectors)[0]
    