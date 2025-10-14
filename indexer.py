import os
import argparse
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

RETRIEVAL_DOCS = int(os.getenv("RETRIEVAL_DOCS"))
RELEVANT_DOCS = int(os.getenv("RELEVANT_DOCS"))

RAW_DOC_PATH = os.getenv("RAW_DOC_PATH")

DEBUG = os.getenv("DEBUG")
DEBUG_MODE = DEBUG.lower() in ("true", "1", "yes", "on") if DEBUG else False


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser(description="A RAG Application.")
    arg_parser.add_argument(
        "--embedprovider",
        type=str,
        default=EMBEDDING_PROVIDER,
        help="The embedding model to use (local, plat or ollama).",
    )
    arg_parser.add_argument(
        "--llmprovider",
        type=str,
        default=LLM_MODEL_PROVIDER,
        help="The vectordb model to use (local or plat).",
    )
    arg_parser.add_argument(
        "--vdbprovider",
        type=str,
        default=VECTORDB_PROVIDER,
        help="The vectordb model to use (local or plat).",
    )
    arg_parser.add_argument(
        "--dbtype",
        type=str,
        default=VECTORDB_TYPE,
        help="The embedding model to use (ollama or openai).",
    )
    arg_parser.add_argument(
        "--root", type=str, default=RAW_DOC_PATH, help="Root dir of docs"
    )
    arg_parser.add_argument("--type", type=str, default=".*", help="Type of doc ext")
    arg_parser.add_argument("--skip", type=str, default=".bak", help="Exclude sub-dirs")
    args = arg_parser.parse_args()
    if args.embedprovider not in SUPPORTED_PROVIDERS.split(","):
        raise ValueError(f"Provider '{args.embedprovider}' not supported.")
    if args.vdbprovider not in SUPPORTED_PROVIDERS.split(","):
        raise ValueError(f"Provider '{args.vdbprovider}' not supported.")
    if args.dbtype not in SUPPORTED_VECTORDBS.split(","):
        raise ValueError(f"VectorDb '{args.dbtype}' not supported.")

    print(
        f"LLM Provider: {args.llmprovider} | LLM model: {LLM_MODEL_NAME} | Embed Provider: {args.embedprovider} | VDB Provider: {args.vdbprovider} | VDB Type: {args.dbtype}"
    )

    target_directory = args.root
    file_extension = args.type
    exclude_subdirs = [args.skip]
    if DEBUG_MODE:
        print(
            f"\ncmd opts: ('from': '{target_directory}', 'match': '{file_extension}', 'skip': {exclude_subdirs}\n"
        )
    if file_extension == "code":
        # target_directory = "./"
        exclude_subdirs.extend(["node_modules", ".*", "dist", ".bak", "venv"])
        desired_extensions = (
            ".c",
            ".cpp",
            ".go",
            ".h",
            ".hpp",
            ".java",
            ".js",
            ".php",
            ".py",
            ".sql",
            ".ts",
            ".tsx",
        )
    else:
        desired_extensions = file_extension
    if DEBUG_MODE:
        print(
            f"after revised: ('match': {desired_extensions}, 'skip': {exclude_subdirs}\n"
        )

    # choose the embedding model
    embedding_model = EmbeddingFactory(
        embedding_provider=args.embedprovider, api_key=EMBEDDING_API_KEY
    )
    embedding_accessor = embedding_model.get_embedding_accessor()

    # choose the vectordb model
    vectordb_model = VectorDbFactory(
        vectordb_provider=args.vdbprovider,
        db_type=args.dbtype,
        api_key=VECTORDB_API_KEY,
    )
    vectordb_accessor = vectordb_model.get_vectordb_accessor()
    vectordb_accessor.set_embedding_function(embedding_accessor)

    files_found = find_files_with_ext(
        target_directory, desired_extensions, exclude_subdirs
    )
    if files_found:
        toal_files = len(files_found)
        for file_path in files_found:
            print(file_path)
        print(f"(Found {toal_files} files in total)")
        print("=" * 80)
    else:
        print("No files match the search conddition ... ")
        exit(0)

    all_chunks, all_vectors, chunks, vectors = (
        [],
        np.array([]),
        [],
        np.array([]),
    )
    for f_id, file in enumerate(files_found):
        try:
            print(f"file({f_id+1} of {toal_files}): {file}")
            _, ext = os.path.splitext(file)
            match ext:
                case ".pdf":
                    chunks = chunk_a_pdf_file(file)
                case ".txt":
                    chunks = chunk_a_text_file(file)
                case _:
                    chunks = chunk_a_code_file(file)
            if DEBUG_MODE:
                file_name = os.path.basename(file)
                print(
                    f"file_prep: chunk_a_{'code' if file_extension == 'code' else 'text'}_file('{file_name}') => chunks({len(chunks)}): {[chunk['line'] for chunk in chunks]}"
                )

            # text_only_chunks = [chunk["text"] for chunk in chunks]
            # vectors = embedding_accessor.embed_documents(text_only_chunks)

            # if DEBUG_MODE:
            #     print(
            #         f"embedding: embed_multi_chunks_{args.embedprovider}('{file_name}') => vectors_shape: {vectors.shape}"
            #     )
            print("-" * 80)
        except Exception as e:
            print(f"An error occurred with file {file}: {e}")
        else:
            all_chunks.extend(chunks)
            # all_vectors = (
            #     np.concatenate((all_vectors, vectors), axis=0)
            #     if len(all_vectors) != 0 and len(vectors) != 0
            #     else vectors
            # )

    # if len(chunks) == 0:
    #     print("No chunks data are populated! Exit.")
    #     exit(2)

    # all_vectors = (
    #     embedding_accessor.embed_documents(all_chunks)
    #     if len(all_vectors) != len(all_chunks)
    #     else all_vectors
    # )

    if DEBUG_MODE:
        print(
            {
                "total_files": toal_files,
                "all_chunks": [chunk["line"] for chunk in all_chunks],
                "total_chunks": len(all_chunks),
            }
        )
        print("=" * 80)

    # vectordb_accessor.store_the_vectors(all_chunks, all_vectors)
    vectordb_accessor.store_the_chunks(all_chunks)
