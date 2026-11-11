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
CLAIMS_PER_PATENT = 3     # claims shown per patent when results are grouped by patent

# Request words removed from the start and end of a question before it is embedded
# (request_words.py). Matched case insensitively as whole words; never in the middle.
REQUEST_POLITE = ["please", "kindly", "can you", "could you", "would you", "will you",
                  "i want to", "i want", "i need to", "i need", "i would like to", "i would like",
                  "i'd like to", "i'd like"]
REQUEST_VERBS = ["find", "show", "give", "list", "get", "return", "retrieve", "fetch", "search for",
                 "search", "look for", "look up", "looking for", "i am looking for", "i'm looking for",
                 "help me find"]
# These verbs are removed even when the question follows them directly ("find spoke wheels").
# The others only before me/us, a determiner, a noun or a link word, so "return spring for a
# valve" stays as typed.
REQUEST_VERBS_ALONE = ["find", "search for", "look for", "looking for", "i am looking for",
                       "i'm looking for", "help me find", "retrieve"]
REQUEST_ME = ["me", "us"]
REQUEST_ASK = ["what are", "what is", "which are", "which is", "what", "which"]   # only before a noun
REQUEST_DETERMINERS = ["all the", "all", "any", "the", "some", "every", "a", "an"]
REQUEST_NOUNS = ["patent claims", "patent claim", "claims", "claim", "patents", "patent",
                 "documents", "results", "inventions", "invention"]
REQUEST_RELATIVE = ["which", "that", "where"]
REQUEST_RELATIVE_VERBS = ["talk", "talks", "discuss", "discusses", "mention", "mentions", "describe",
                          "describes", "cover", "covers", "deal", "deals", "relate", "relates",
                          "involve", "involves", "contain", "contains", "disclose", "discloses",
                          "include", "includes", "explain", "explains", "show", "shows", "have",
                          "has", "are", "is", "describing", "mentioning", "covering", "discussing",
                          "involving", "containing", "disclosing", "including"]
REQUEST_LINKS = ["about", "on", "regarding", "concerning", "related to", "relating to", "for", "of",
                 "with", "to"]
REQUEST_TRAILING = ["please", "thanks", "thank you"]

# Web server
HOST = "127.0.0.1"
PORT = 8000

# Evaluation (python -m patent_search eval)
EVAL_QUESTIONS_PATH = PROJECT_ROOT / "tests" / "eval" / "questions.json"
EVAL_RESULTS_DIR = Path(os.environ.get("PATENT_SEARCH_EVAL_RESULTS_DIR", PROJECT_ROOT / "data" / "eval_results"))
EVAL_CACHE_DIR = Path(os.environ.get("PATENT_SEARCH_EVAL_CACHE_DIR", PROJECT_ROOT / "data" / "eval_cache"))
EVAL_JUDGMENTS_DIR = Path(os.environ.get("PATENT_SEARCH_EVAL_JUDGMENTS_DIR", PROJECT_ROOT / "data" / "eval_judgments"))
EVAL_TOP_N = 10
EVAL_BATCH_SIZE = 32
# Grades B3 will give each claim per question: 0 not relevant, 1 partly, 2 fully.
EVAL_RELEVANT_GRADE = 1      # counts for Hit@5, P@5, Recall@10, MRR
EVAL_FULL_GRADE = 2          # "fully relevant" for the first-full-rank metric

# BM25 word-matching baseline
BM25_K1 = 1.5
BM25_B = 0.75

# Embedding models to compare. "nomic" is the production model and reuses data/index.
# Optional per model: batch_size (default EVAL_BATCH_SIZE) and max_seq_length (default: the
# model's own limit).
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
# Qwen's own format: an instruction, a real line break, then "Query:" with no space after it.
QWEN_QUERY_PREFIX = "Instruct: Given a patent search question, retrieve patent claims that answer it\nQuery:"
EVAL_MODELS = {
    "nomic": {
        "model_name": MODEL_NAME, "revision": MODEL_REVISION, "code_revision": MODEL_CODE_REVISION,
        "query_prefix": QUERY_PREFIX, "document_prefix": DOCUMENT_PREFIX, "trust_remote_code": True,
        "reuse_index": True,
    },
    "bge-small": {
        "model_name": "BAAI/bge-small-en-v1.5", "revision": "5c38ec7c405ec4b44b94cc5a9bb96e735b38267a",
        "query_prefix": BGE_QUERY_PREFIX, "document_prefix": "",
    },
    "bge-base": {
        "model_name": "BAAI/bge-base-en-v1.5", "revision": "a5beb1e3e68b9ab74eb54cfd186867f64f240e1a",
        "query_prefix": BGE_QUERY_PREFIX, "document_prefix": "",
    },
    "patentsberta": {
        "model_name": "AI-Growth-Lab/PatentSBERTa", "revision": "3ff1d553c861d8f5bfd902333d97fc95eb6b4c8f",
        "query_prefix": "", "document_prefix": "",
    },
    "minilm": {
        "model_name": "sentence-transformers/all-MiniLM-L6-v2", "revision": "1110a243fdf4706b3f48f1d95db1a4f5529b4d41",
        "query_prefix": "", "document_prefix": "",
    },
    # 0.6B parameters: batches of 16 overflow 16 GB on mps, so 4. Capped at nomic's 8,192 tokens
    # (its own limit is 131,072); the longest text here is 1,151 Qwen tokens, so nothing is cut.
    "qwen3-0.6b": {
        "model_name": "Qwen/Qwen3-Embedding-0.6B", "revision": "97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3",
        "query_prefix": QWEN_QUERY_PREFIX, "document_prefix": "", "batch_size": 4, "max_seq_length": 8192,
    },
}
