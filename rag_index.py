import os
import numpy as np
from dotenv import load_dotenv
from utils.doc_file_find import find_files_with_ext
from utils import chunk_a_text_file, chunk_a_code_file, chunk_a_pdf_file
from plat.embedding.embedding_factory import EmbeddingFactory
from plat.vectordb.vectordb_factory import VectorDbFactory

load_dotenv()
ENV_PATH = ".env"
SUPPORTED_PROVIDERS = os.getenv("SUPPORTED_PROVIDERS")

EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER")
EMBEDDING_API_URL = os.getenv("EMBEDDING_API_URL")
EMBEDDING_API_KEY = os.getenv("EMBEDDING_API_KEY")

LLM_MODEL_PROVIDER = os.getenv("LLM_MODEL_PROVIDER")
LLM_MODEL_NAME = os.getenv("LLM_MODEL_NAME")
LLM_MODEL_API_URL = os.getenv("LLM_MODEL_API_URL")
LLM_MODEL_API_KEY = os.getenv("LLM_MODEL_API_KEY")

SUPPORTED_VECTORDBS = os.getenv("SUPPORTED_VECTORDBS")
VECTORDB_PROVIDER = os.getenv("VECTORDB_PROVIDER")
VECTORDB_TYPE = os.getenv("VECTORDB_TYPE")
VECTORDB_API_URL = os.getenv("LLM_MODEL_API_URL")
VECTORDB_API_KEY = os.getenv("LLM_MODEL_API_KEY")
VECTORDB_ROOT = ".vdb"

# RETRIEVAL_DOCS = int(os.getenv("RETRIEVAL_DOCS"))
# RELEVANT_DOCS = int(os.getenv("RELEVANT_DOCS"))

RAW_DOC_PATH = os.getenv("RAW_DOC_PATH")

# DEBUG = os.getenv("DEBUG")
# DEBUG_MODE = DEBUG.lower() in ("true", "1", "yes", "on") if DEBUG else False


def docIndex():
    target_directory = RAW_DOC_PATH
    exclude_subdirs = [".bak"]
    desired_extensions = ".*"

    # choose the embedding model
    embedding_model = EmbeddingFactory(
        embedding_provider=EMBEDDING_PROVIDER, api_key=EMBEDDING_API_KEY
    )
    embedding_accessor = embedding_model.get_embedding_accessor()

    # choose the vectordb model
    vectordb_model = VectorDbFactory(
        vectordb_provider=VECTORDB_PROVIDER,
        db_type=VECTORDB_TYPE,
        api_key=VECTORDB_API_KEY,
    )
    vectordb_accessor = vectordb_model.get_vectordb_accessor()
    vectordb_accessor.set_embedding_function(embedding_accessor)

    # find files in the raw_doc stored
    files_found = find_files_with_ext(
        target_directory, desired_extensions, exclude_subdirs
    )
    if files_found:
        toal_files = len(files_found)
        for file_path in files_found:
            print(file_path)
        print(f"(Found {toal_files} files in total)")
    else:
        print("No files match the search conddition ... ")
        exit(0)

    # chunk all found files
    all_chunks, all_vectors, chunks, vectors = (
        [],
        np.array([]),
        [],
        np.array([]),
    )
    for f_id, file in enumerate(files_found):
        try:
            _, ext = os.path.splitext(file)
            match ext:
                case ".pdf":
                    chunks = chunk_a_pdf_file(file)
                case ".txt":
                    chunks = chunk_a_text_file(file)
                case _:
                    chunks = chunk_a_code_file(file)
        except Exception as e:
            print(f"An error occurred with file {file}: {e}")
        else:
            all_chunks.extend(chunks)

    # index & save all chunks
    vectordb_accessor.store_the_chunks(all_chunks)


# if __name__ == "__main__":
#     print(
#         f"LLM Provider: {LLM_MODEL_PROVIDER} | LLM model: {LLM_MODEL_NAME} | Embed Provider: {EMBEDDING_PROVIDER} | VDB Provider: {VECTORDB_PROVIDER} | VDB Type: {VECTORDB_TYPE}"
#     )
#     docIndex()