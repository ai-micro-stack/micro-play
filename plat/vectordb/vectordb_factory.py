from plat.vectordb.vectordb_chroma import PlatServedChromaDb
from plat.vectordb.vectordb_faiss import PlatServedFaissDb


class VectorDbFactory:
    def __init__(self, vectordb_provider: str, db_type: str = "faiss", api_key: str = None):
        self.vectordb_provider = vectordb_provider
        self.db_type = db_type
        self.api_url = "api_url"
        self.api_key = api_key

    def get_vectordb_accessor(self):
        if self.db_type == "faiss":
            return PlatServedFaissDb(
                vectordb_provider = self.vectordb_provider, api_url = self.api_url, api_key = self.api_key
            )
        elif self.db_type == "chroma":
            return PlatServedChromaDb(
                vectordb_provider = self.vectordb_provider, api_url = self.api_url, api_key = self.api_key
            )
        else:
            raise ValueError(f"Unsupported model type: {self.db_type}")
