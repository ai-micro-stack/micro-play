import os
import json
import argparse
from dotenv import load_dotenv
from plat.vectordb.vectordb_factory import VectorDbFactory

# from plat.vectordb import convert_index_to_tsv

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

DEBUG = os.getenv("DEBUG")
DEBUG_MODE = DEBUG.lower() in ("true", "1", "yes", "on") if DEBUG else False

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
arg_parser.add_argument("--full", type=str, default="No", help="Output Full Content")
args = arg_parser.parse_args()
if args.vdbprovider not in SUPPORTED_PROVIDERS.split(","):
    raise ValueError(f"Provider '{args.vdbprovider}' not supported.")

full_data = args.full.lower() in ("true", "1", "yes", "on") if args.full else False

print(
    f"LLM Provider: {args.llmprovider} | LLM model: {LLM_MODEL_NAME} | Embed Provider: {args.embedprovider} | VDB Provider: {args.vdbprovider} | VDB Type: {args.dbtype}"
)

# choose the vectordb model
vectordb_model = VectorDbFactory(
    vectordb_provider=args.vdbprovider, db_type=args.dbtype, api_key=None
)
vectordb_accessor = vectordb_model.get_vectordb_accessor()

result = vectordb_accessor.convert_index_to_tsv(full_data=full_data)
print("You can view your index 3D projection at projector.tensorflow.orgs:")
print("### Don't upload any confidential data! ###")
print(json.dumps(result, indent=4))
