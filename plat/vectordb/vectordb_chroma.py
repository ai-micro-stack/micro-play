import os
from langchain_chroma import Chroma
from langchain_core.documents import Document
from dotenv import load_dotenv


load_dotenv()
# VECTORDB_PROVIDER = os.getenv("VECTORDB_PROVIDER")
VECTORDB_ROOT = os.getenv("VECTORDB_ROOT")  # default '.vdb'
# VECTORDB_API_URL = os.getenv("VECTORDB_API_URL")
# VECTORDB_API_KEY = os.getenv("VECTORDB_API_KEY")


class PlatServedChromaDb:
    def __init__(self, vectordb_provider: str, api_url: str, api_key: str = None):
        # self.vectordb_provider = vectordb_provider
        self.api_url = api_url
        self.api_key = api_key
        self.vectordb_model = Chroma
        self.embedding_path = os.path.join(
            VECTORDB_ROOT, "chroma-" + vectordb_provider, "index.embed"
        )
        self.embedding_function = None  # load embedding from disk here
        self.vectordb_path = os.path.join(
            VECTORDB_ROOT, "chroma-" + vectordb_provider, "index.chroma"
        )
        os.makedirs(self.vectordb_path, exist_ok=True)
        self.vectorstore = None  ## Chroma.get_or_create_collection(name=self.vectordb_path, embedding_fuction=self.embedding_function)

    def set_embedding_function(self, embedding_fuction):
        self.embedding_function = embedding_fuction
        self.vectorstore = Chroma(
            collection_name="my_persistent_collection",
            embedding_function=self.embedding_function,
            persist_directory=self.vectordb_path,
        )

    def store_the_chunks(self, chunks):
        documents, ids = [], []
        for chunk in chunks:
            metadata = {"source": chunk["id"], "file": chunk["file"], "page": chunk["page"], "line": chunk["line"], "count": chunk["count"]}
            documents.append(
                Document(page_content=chunk["text"], metadata=metadata)
            )
            ids.append(chunk["id"])
        self.vectorstore.add_documents(
            documents=documents,
            ids=ids,
        )

    def search_similar_chunks(self, query_text, k=5):
        results = self.vectorstore.similarity_search_with_score(query=query_text, k=k)
        return results
