import os
import json
import requests
import numpy as np
from dotenv import load_dotenv

load_dotenv()
# EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER")
EMBEDDING_API_URL = os.getenv("EMBEDDING_API_URL")
# VECTORDB_API_KEY = os.getenv("VECTORDB_API_KEY")


class PlatServedEmbeddings:
    """
    class that implements two methods to be called from Chroma
    """

    def __init__(self, model: str):
        # load_dotenv()
        self.model = model
        self.embedding_api_url = EMBEDDING_API_URL
        self.api_endpoint = self.embedding_api_url + "/api/embed"

    def embed_documents(self, texts: list[str]):
        payload = {"model": self.model, "input": texts}
        response = requests.post(self.api_endpoint, json=payload)
        vectors = response.json()["embeddings"]
        return np.array(vectors)

    def embed_query(self, text: str):
        headers = {"Content-Type": "application/json"}
        payload = {"model": self.model, "input": text}
        response = requests.post(
            self.api_endpoint, data=json.dumps(payload), headers=headers
        )
        vectors = response.json()["embeddings"]
        return vectors[0]
