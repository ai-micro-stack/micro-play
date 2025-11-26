import requests
from config import config


class OllamaEmbeddings:
    """
    Pure Python implementation of Ollama embeddings without LangChain.
    """

    def __init__(self, model: str):
        self.model = model
        self.api_url = config.EMBEDDING_API_URL.rstrip('/') + "/api/embed"

    def embed_documents(self, texts: list[str]):
        """Embed multiple documents."""
        embeddings = []
        for text in texts:
            embedding = self.embed_query(text)
            embeddings.append(embedding)
        return embeddings

    def embed_query(self, text: str):
        """Embed a single query."""
        payload = {
            "model": self.model,
            "input": text
        }

        try:
            response = requests.post(self.api_url, json=payload, timeout=config.OLLAMA_EMBEDDING_TIMEOUT)
            response.raise_for_status()
            result = response.json()
            return result["embeddings"][0]
        except requests.exceptions.RequestException as e:
            raise Exception(f"Ollama embedding API request failed: {e}")
        except (KeyError, IndexError) as e:
            raise Exception(f"Invalid response from Ollama embedding API: {e}")