import os
import faiss
import pickle
import numpy as np
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_core.documents import Document
from dotenv import load_dotenv


load_dotenv()
# VECTORDB_PROVIDER = os.getenv("VECTORDB_PROVIDER")
VECTORDB_ROOT = os.getenv("VECTORDB_ROOT")  # default '.vdb'
# VECTORDB_API_URL = os.getenv("VECTORDB_API_URL")
# VECTORDB_API_KEY = os.getenv("VECTORDB_API_KEY")
# os.environ["OPENAI_API_KEY"] = "YOUR_OPENAI_API_KEY"


class PlatServedFaissDb:
    def __init__(self, vectordb_provider: str, api_url: str, api_key: str = None):
        # self.vectordb_provider = vectordb_provider
        self.api_url = api_url
        self.api_key = api_key
        self.vectordb_model = FAISS
        self.embedding_path = os.path.join(
            VECTORDB_ROOT, "faiss-" + vectordb_provider, "index.embed"
        )
        self.embedding_function = None  # load embedding from disk here
        self.vectordb_path = os.path.join(
            VECTORDB_ROOT, "faiss-" + vectordb_provider, "index.faiss"
        )
        self.faiss_index = None
        os.makedirs(self.vectordb_path, exist_ok=True)
        self.vectorstore = None  # self.vectordb_model.load_local(self.vectordb_path, self.embedding_function.embed_query, allow_dangerous_deserialization = True) if os.path.exists(self.vectordb_path) else None
        self.index_dt_path = os.path.join(
            VECTORDB_ROOT, "faiss-" + vectordb_provider, "index.tsv"
        )

    def set_embedding_function(self, embedding_fuction):
        self.embedding_function = embedding_fuction
        embedding_dim = len(embedding_fuction.embed_query("hello world"))
        self.faiss_index = faiss.IndexFlatL2(embedding_dim)
        if os.path.isfile(os.path.join(self.vectordb_path, "index.faiss")):
            self.vectorstore = self.vectordb_model.load_local(
                self.vectordb_path,
                self.embedding_function.embed_query,
                allow_dangerous_deserialization=True,
            )
        else:
            self.vectorstore = FAISS(
                embedding_function=embedding_fuction,
                index=self.faiss_index,
                docstore= InMemoryDocstore(),
                index_to_docstore_id={}
            )
        # persist embeding function to disk here

    def store_the_chunks(self, chunks):
        documents, ids = [], []
        for chunk in chunks:
            metadata = {"source": chunk["id"], "file": chunk["file"], "page": chunk["page"], "line": chunk["line"], "count": chunk["count"]}
            documents.append(
                Document(page_content=chunk["text"], metadata=metadata)
            )
            ids.append(chunk["id"])
        # self.vectorStore.add_documents(documents)
        self.vectorstore = self.vectordb_model.from_documents(
            documents, self.embedding_function
        )
        self.vectorstore.save_local(self.vectordb_path)

    def search_similar_chunks(self, query_text, k=5):
        results = self.vectorstore.similarity_search_with_score(query_text, k=k)
        return results

    def convert_index_to_tsv(self, full_data=False):
        index = faiss.read_index(os.path.join(self.vectordb_path, "index.faiss"))
        vectors = index.reconstruct_n(0, index.ntotal)
        np.savetxt(self.index_dt_path, vectors, delimiter="\t")

        with open(os.path.join(self.vectordb_path, "index.pkl"), "rb") as f:
            chunks = pickle.load(f)
        print(chunks)

        return {
            "projector-url": "https://projector.tensorflow.org/",
            "vectors-tsv": self.index_dt_path,
            # "metadata-tsv": self.chunk_dt_path,
        }
