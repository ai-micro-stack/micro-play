"""
Document indexing module for the RAG system.

This module handles the processing and indexing of documents into the vector database.
It discovers files, chunks them into semantic units, generates embeddings, and stores
them for later retrieval.
"""

import os
import numpy as np
from utils.doc_file_find import find_files_with_ext
from utils import chunk_a_text_file, chunk_a_code_file, chunk_a_pdf_file
from plat.embedding.embedding_factory import EmbeddingFactory
from plat.vectordb.vectordb_factory import VectorDbFactory
from config import config


def docIndex():
    target_directory = config.RAW_DOC_PATH
    exclude_subdirs = [".bak"]
    desired_extensions = ".*"

    # choose the embedding model
    embedding_model = EmbeddingFactory(
        embedding_provider=config.EMBEDDING_PROVIDER, api_key=config.EMBEDDING_API_KEY
    )
    embedding_accessor = embedding_model.get_embedding_accessor()

    # choose the vectordb model
    vectordb_model = VectorDbFactory(
        vectordb_provider=config.VECTORDB_PROVIDER,
        db_type=config.VECTORDB_TYPE,
        api_key=config.VECTORDB_API_KEY,
    )
    vectordb_accessor = vectordb_model.get_vectordb_accessor()
    vectordb_accessor.set_embedding_function(embedding_accessor)

    # find files in the raw_doc stored
    files_found = find_files_with_ext(
        target_directory, desired_extensions, exclude_subdirs
    )
    files_new = []
    if files_found:
        for file_path in files_found:
            is_indexed=vectordb_accessor.check_file_is_indexed(file_path)
            if is_indexed:
                continue
            files_new.append(file_path)
        total_files = len(files_found)
        total_new = len(files_new)
        if total_new == 0:
            return
    else:
        return

    # chunk all found files & load embeddings in vector store
    all_chunks, all_vectors, chunks, vectors = (
        [],
        np.array([]),
        [],
        np.array([]),
    )
    for f_id, file in enumerate(files_new):
        try:
            _, ext = os.path.splitext(file)
            match ext:
                case ".pdf":
                    chunks = chunk_a_pdf_file(file, config.MAX_CHUNK_SIZE, config.CHUNK_OVERLAP)
                case ".txt":
                    chunks = chunk_a_text_file(file, config.MAX_CHUNK_SIZE, config.CHUNK_OVERLAP)
                case _:
                    chunks = chunk_a_code_file(file, config.MAX_CHUNK_SIZE, config.CHUNK_OVERLAP)
        except Exception as e:
            print(f"An error occurred with file {file}: {e}")
        else:
            vectordb_accessor.store_the_chunks(chunks)

    # persist the vector store
    vectordb_accessor.persist_vector_store()