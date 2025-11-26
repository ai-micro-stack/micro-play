"""
Centralized configuration management for the RAG application.
Handles all environment variables and provides a single source of truth for configuration.
"""
import os
from typing import Optional
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class Config:
    """Centralized configuration class for the RAG application."""

    # Debug settings
    DEBUG: bool = os.getenv("DEBUG", "false").lower() in ("true", "1", "yes", "on")

    # Supported providers
    SUPPORTED_PROVIDERS: str = os.getenv("SUPPORTED_PROVIDERS", "local,plat,ollama,openai,bedrock")
    SUPPORTED_VECTORDBS: str = os.getenv("SUPPORTED_VECTORDBS", "faiss,chroma,milvus")

    # Embedding configuration
    EMBEDDING_PROVIDER: str = os.getenv("EMBEDDING_PROVIDER", "plat")
    EMBEDDING_MODEL_NAME: str = os.getenv("EMBEDDING_MODEL_NAME", "mahonzhan/all-MiniLM-L6-v2")
    EMBEDDING_API_URL: str = os.getenv("EMBEDDING_API_URL", "http://localhost:11434")
    EMBEDDING_API_KEY: Optional[str] = os.getenv("EMBEDDING_API_KEY")

    # LLM configuration
    LLM_MODEL_PROVIDER: str = os.getenv("LLM_MODEL_PROVIDER", "plat")
    LLM_MODEL_NAME: str = os.getenv("LLM_MODEL_NAME", "gemma3:270m")
    LLM_MODEL_API_URL: str = os.getenv("LLM_MODEL_API_URL", "http://localhost:11434")
    LLM_MODEL_API_KEY: Optional[str] = os.getenv("LLM_MODEL_API_KEY")

    # OpenAI and Anthropic API keys
    OPENAI_API_KEY: Optional[str] = os.getenv("OPENAI_API_KEY")
    ANTHROPIC_API_KEY: Optional[str] = os.getenv("ANTHROPIC_API_KEY")

    # Vector database configuration
    VECTORDB_PROVIDER: str = os.getenv("VECTORDB_PROVIDER", "local")
    VECTORDB_TYPE: str = os.getenv("VECTORDB_TYPE", "faiss")
    VECTORDB_API_URL: str = os.getenv("VECTORDB_API_URL", "http://localhost:8000")
    VECTORDB_API_KEY: Optional[str] = os.getenv("VECTORDB_API_KEY")
    VECTORDB_ROOT: str = os.getenv("VECTORDB_ROOT", ".vdb")

    # Retrieval configuration
    RETRIEVAL_DOCS: int = int(os.getenv("RETRIEVAL_DOCS", "9"))
    RELEVANT_DOCS: int = int(os.getenv("RELEVANT_DOCS", "3"))

    # Document storage
    RAW_DOC_PATH: str = os.getenv("RAW_DOC_PATH", "raw_docs")

    # Chunking configuration
    MAX_CHUNK_SIZE: int = int(os.getenv("MAX_CHUNK_SIZE", "500"))
    CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "50"))

    # Reranking configuration
    RERANK_METHOD: str = os.getenv("RERANK_METHOD", "cross_encoder")
    RERANK_MODEL: str = os.getenv("RERANK_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
    DIVERSITY_WEIGHT: float = float(os.getenv("DIVERSITY_WEIGHT", "0.3"))
    RERANK_BATCH_SIZE: int = int(os.getenv("RERANK_BATCH_SIZE", "32"))

    # API timeouts (in seconds)
    OLLAMA_LLM_TIMEOUT: int = int(os.getenv("OLLAMA_LLM_TIMEOUT", "300"))  # 5 minutes for LLM calls
    OLLAMA_EMBEDDING_TIMEOUT: int = int(os.getenv("OLLAMA_EMBEDDING_TIMEOUT", "120"))  # 2 minutes for embeddings

    @classmethod
    def validate_providers(cls) -> None:
        """Validate that configured providers are supported."""
        if cls.EMBEDDING_PROVIDER not in cls.SUPPORTED_PROVIDERS.split(","):
            raise ValueError(f"Unsupported embedding provider: {cls.EMBEDDING_PROVIDER}")
        if cls.LLM_MODEL_PROVIDER not in cls.SUPPORTED_PROVIDERS.split(","):
            raise ValueError(f"Unsupported LLM provider: {cls.LLM_MODEL_PROVIDER}")
        if cls.VECTORDB_PROVIDER not in cls.SUPPORTED_PROVIDERS.split(","):
            raise ValueError(f"Unsupported vector DB provider: {cls.VECTORDB_PROVIDER}")
        if cls.VECTORDB_TYPE not in cls.SUPPORTED_VECTORDBS.split(","):
            raise ValueError(f"Unsupported vector DB type: {cls.VECTORDB_TYPE}")

    @classmethod
    def validate_api_keys(cls) -> None:
        """Validate that required API keys are present for cloud providers."""
        if cls.EMBEDDING_PROVIDER == "openai" and not cls.EMBEDDING_API_KEY:
            raise ValueError("OpenAI API key required for OpenAI embeddings")
        if cls.LLM_MODEL_PROVIDER == "gpt" and not cls.OPENAI_API_KEY:
            raise ValueError("OpenAI API key required for GPT models")
        if cls.LLM_MODEL_PROVIDER == "claude" and not cls.ANTHROPIC_API_KEY:
            raise ValueError("Anthropic API key required for Claude models")

# Global config instance
config = Config()

# Validate configuration on import
try:
    config.validate_providers()
    config.validate_api_keys()
except ValueError as e:
    print(f"Configuration error: {e}")
    if not config.DEBUG:
        raise