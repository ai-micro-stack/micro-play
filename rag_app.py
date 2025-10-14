import os
import argparse
from plat.vectordb.vectordb_factory import VectorDbFactory
from plat.llmodel.llmodel_factory import LLModelFactory
from plat.embedding.embedding_factory import EmbeddingFactory
from rerank.rerank_retrieved_docs import get_context_from_documents
from dotenv import load_dotenv, set_key

load_dotenv()
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
    args = arg_parser.parse_args()
    if args.embedprovider not in SUPPORTED_PROVIDERS.split(","):
        raise ValueError(f"Provider '{args.embedprovider}' not supported.")
    if args.llmprovider not in SUPPORTED_PROVIDERS.split(","):
        raise ValueError(f"Provider '{args.llmprovider}' not supported.")
    if args.vdbprovider not in SUPPORTED_PROVIDERS.split(","):
        raise ValueError(f"Provider '{args.vdbprovider}' not supported.")
    if args.dbtype not in SUPPORTED_VECTORDBS.split(","):
        raise ValueError(f"VectorDb '{args.dbtype}' not supported.")

    print(
        f"LLM Provider: {args.llmprovider} | LLM model: {LLM_MODEL_NAME} | Embed Provider: {args.embedprovider} | VDB Provider: {args.vdbprovider} | VDB Type: {args.dbtype}"
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
    print(f"VectorDb: {vectordb_accessor.vectordb_path}")
    vectordb_accessor.set_embedding_function(embedding_accessor)

    # choose the llm model
    llm_model = LLModelFactory(
        llmodel_provider=args.llmprovider,
        model_name="gemma3:270m",
        api_key=None,
    )
    llmodel_accessor = llm_model.get_llmodel_accessor()

    while True:
        query_text = input("\nAsk a question ('/bye' to quit): ")
        if query_text.lower() == "/bye":
            break

        # Retrieve and rerank the results
        results = vectordb_accessor.search_similar_chunks(
            query_text, RETRIEVAL_DOCS
        )
        enhanced_context_text, sources = get_context_from_documents(results, 3)

        # Generate the response from LLM
        answer = llmodel_accessor.generate_response(enhanced_context_text, query_text)
        print("\nAnswer from RAG:\n", answer)
        print("Reference Doc Snippets:")
        print("\n".join(sources))
