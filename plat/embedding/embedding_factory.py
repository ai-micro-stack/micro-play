from plat.embedding.embeddings_openai import OpenAIEmbeddings
from plat.embedding.embeddings_huggingface import SentenceTransformerEmbeddings
from plat.embedding.embeddings_microplat import PlatServedEmbeddings
from plat.embedding.embeddings_ollama import OllamaEmbeddings
from plat.embedding.embeddings_bedrock import BedrockEmbeddings
from config import config

class EmbeddingFactory:
    def __init__(self, embedding_provider: str, api_key: str = None):
        self.embedding_provider = embedding_provider
        self.api_key = api_key

    def get_embedding_accessor(self):
        match self.embedding_provider:
            case "local":
                return SentenceTransformerEmbeddings(model=config.EMBEDDING_MODEL_NAME)
            case "plat":
                return PlatServedEmbeddings(model=config.EMBEDDING_MODEL_NAME)
            case "ollama":
                return OllamaEmbeddings(model=config.EMBEDDING_MODEL_NAME)
            case "openai":
                if not self.api_key:
                    raise ValueError(
                        "OpenAI API key must be provided for OpenAI embeddings"
                    )
                return OpenAIEmbeddings(api_key=self.api_key)
            case "bedrock":
                return BedrockEmbeddings(
                    credentials_profile_name="default", region_name="us-east-1"
                )
            case _:
                raise ValueError(
                    f"Unsupported embedding model: {self.embedding_provider}"
                )
