"""All tunable values. Paths can be overridden with environment variables."""

import os
from pathlib import Path

# Paths
PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = Path(os.environ.get("PATENT_SEARCH_RAW_DIR", PROJECT_ROOT / "data 2" / "patent_data_small"))
RAW_GLOB = "patents_*.json"
INDEX_DIR = Path(os.environ.get("PATENT_SEARCH_INDEX_DIR", PROJECT_ROOT / "data" / "index"))
DB_PATH = INDEX_DIR / "search.db"
CLAIM_VECTORS_PATH = INDEX_DIR / "claim_vectors.npy"
PATENT_VECTORS_PATH = INDEX_DIR / "patent_vectors.npy"
META_PATH = INDEX_DIR / "meta.json"

# Embedding model
MODEL_NAME = "nomic-ai/nomic-embed-text-v1.5"
MODEL_REVISION = "e9b6763023c676ca8431644204f50c2b100d9aab"
# The model loads its code from nomic-ai/nomic-bert-2048; pin that too.
MODEL_CODE_REVISION = "7710840340a098cfb869c4f65e87cf2b1b70caca"
VECTOR_SIZE = 768
DOCUMENT_PREFIX = "search_document: "
QUERY_PREFIX = "search_query: "
EMBED_BATCH_SIZE = 16

# Claim parsing
PARENT_SEARCH_CHARS = 200

# Search
W_CLAIM = 0.8
W_PATENT = 0.2
TOP_N_DEFAULT = 10
TOP_N_MAX = 100

# Web server
HOST = "127.0.0.1"
PORT = 8000
