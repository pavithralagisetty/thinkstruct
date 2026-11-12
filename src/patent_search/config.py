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
NOMIC_MODEL_NAME = "nomic-ai/nomic-embed-text-v1.5"
NOMIC_MODEL_REVISION = "e9b6763023c676ca8431644204f50c2b100d9aab"
# The model loads its code from nomic-ai/nomic-bert-2048; pin that too.
NOMIC_CODE_REVISION = "7710840340a098cfb869c4f65e87cf2b1b70caca"
# PATENT_SEARCH_MODEL points search at another model folder, for example the Part 3 fine tuned
# model (models/nomic-finetuned). A local folder has no Hub revision to pin. The index must be
# rebuilt with the same model; search refuses a mismatch.
_MODEL_OVERRIDE = os.environ.get("PATENT_SEARCH_MODEL")
MODEL_NAME = _MODEL_OVERRIDE or NOMIC_MODEL_NAME
MODEL_REVISION = None if _MODEL_OVERRIDE else NOMIC_MODEL_REVISION
MODEL_CODE_REVISION = None if _MODEL_OVERRIDE else NOMIC_CODE_REVISION
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

# Patent map (landscape.py): t-SNE of the patent vectors, cached next to the index
MAP_SEED = 42
MAP_PERPLEXITY = 30

# Part 3: fine tune nomic on (query, first claim, wrong first claim) rows (training.py)
PART3_RESULTS_DIR = PROJECT_ROOT / "results"
PART3_MODEL_OUT = PROJECT_ROOT / "models" / "nomic-finetuned"
PART3_CHECKPOINT_DIR = PROJECT_ROOT / "models" / ".part3_checkpoints"
PART3_QUERY_FIELDS = ("abstract", "title")
PART3_MAX_SEQ_LENGTH = 512
PART3_TRAIN_FRACTION = 0.8          # oldest 80% of weekly files train, newest 20% test
PART3_CLASS_PREFIX_LEN = 4          # wrong answers share this classification prefix (B60B)
PART3_SEED = 42
PART3_EPOCHS = 1
PART3_LEARNING_RATE = 2e-5
PART3_WARMUP_RATIO = 0.1
PART3_BATCH_SIZE = 4             # 8 and 16 run out of memory on mps (16 GB Mac)
PART3_EVAL_BATCH_SIZE = 16
PART3_RECALL_CEILING = 0.95         # base Recall@10 at or above this: note it, rerun with title

# Part 3, second fine tune (train_summary.py): question = the patent's 1 to 2 sentence summary
# (ADAPTER_SUMMARIES_PATH), right answer = one of its claims' stored embed_text (dependent claims
# included), wrong answer = a claim of another patent in the same CPC main group, else the same
# subclass, else any patent, from the same split. Same split, learning rate, epochs and batch
# size as the first fine tune.
PART3S_MODEL_OUT = PROJECT_ROOT / "models" / "nomic-finetuned-summary"
PART3S_CLAIMS_PER_PATENT = 5        # the main claim plus up to 4 more, so long patents do not dominate
PART3S_FALSE_NEGATIVE_RATIO = 0.9   # skip a wrong claim that base nomic scores >= 90% of the right one

# Part 3 query adapter (adapter.py): a 768 x 768 layer on the question vector, trained so an
# abstract scores its own patent's claims above other patents' claims. The model and the stored
# claim vectors do not change. Starts as the identity, so untrained it changes nothing.
ADAPTER_PATH = PROJECT_ROOT / "models" / "query_adapter.npz"
ADAPTER_ABSTRACT_CACHE = PROJECT_ROOT / "data" / "eval_cache" / "adapter_abstracts.npz"
# Search uses the adapter only when this is set (path to an adapter file); unset, search is unchanged.
QUERY_ADAPTER = os.environ.get("PATENT_SEARCH_QUERY_ADAPTER") or None
ADAPTER_VAL_FRACTION = 0.15         # newest train weeks pick the epoch; the test weeks stay unseen
ADAPTER_TEMPERATURE = 0.05          # softmax temperature over cosine scores
ADAPTER_LEARNING_RATE = 1e-4         # picked on the validation weeks (1e-3 overshoots by epoch 20)
ADAPTER_EPOCHS = 300                # full batch: every train abstract against every train claim
ADAPTER_IDENTITY_PENALTY = 1.0      # weight of ||W - I||^2 + ||b||^2: keeps the layer near "no change"
ADAPTER_SEED = 42
# What the adapter trains on as the query. "abstract": the patent's abstract. "summary": a 1 to 2
# sentence plain-English summary of the abstract (data/part3_summaries/summaries.json, written
# from the abstract alone), which is closer to a short search question. Each source has its own
# query vector cache, adapter file and results files.
ADAPTER_SUMMARIES_PATH = PROJECT_ROOT / "data" / "part3_summaries" / "summaries.json"
ADAPTER_SOURCES = {
    "abstract": {"cache": ADAPTER_ABSTRACT_CACHE, "out": ADAPTER_PATH, "results": "adapter_results"},
    "summary": {"cache": PROJECT_ROOT / "data" / "eval_cache" / "adapter_summaries.npz",
                "out": PROJECT_ROOT / "models" / "query_adapter_summary.npz",
                "results": "adapter_summary_results"},
}

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
    # Part 3 fine tuned nomic (training.py). A local folder has no revision, so the cache key
    # holds a hash of its weights instead. It trained at 512 tokens; read 8,192 like nomic.
    "nomic-finetuned": {
        "model_name": str(PART3_MODEL_OUT), "revision": None,
        "query_prefix": QUERY_PREFIX, "document_prefix": DOCUMENT_PREFIX, "trust_remote_code": True,
        "batch_size": 8, "max_seq_length": 8192,
    },
    # The second Part 3 fine tune: summaries as questions, every kind of claim as answers.
    "nomic-finetuned-summary": {
        "model_name": str(PART3S_MODEL_OUT), "revision": None,
        "query_prefix": QUERY_PREFIX, "document_prefix": DOCUMENT_PREFIX, "trust_remote_code": True,
        "batch_size": 8, "max_seq_length": 8192,
    },
    # Production nomic with the Part 3 query adapter on the question vector (adapter.py). Same
    # claim and patent vectors as nomic (data/index); only the question vector changes.
    "nomic-adapter": {
        "model_name": MODEL_NAME, "revision": MODEL_REVISION, "code_revision": MODEL_CODE_REVISION,
        "query_prefix": QUERY_PREFIX, "document_prefix": DOCUMENT_PREFIX, "trust_remote_code": True,
        "reuse_index": True, "query_adapter": ADAPTER_PATH,
    },
    # The same, with the adapter trained on 1 to 2 sentence summaries instead of abstracts.
    "nomic-adapter-summary": {
        "model_name": MODEL_NAME, "revision": MODEL_REVISION, "code_revision": MODEL_CODE_REVISION,
        "query_prefix": QUERY_PREFIX, "document_prefix": DOCUMENT_PREFIX, "trust_remote_code": True,
        "reuse_index": True, "query_adapter": ADAPTER_SOURCES["summary"]["out"],
    },
    # 4B parameters, 2,560 numbers per vector: too large for this Mac, so every vector, the 40
    # questions included, is embedded on a Colab GPU (data/eval_cache/colab/colab_qwen4b.py) and
    # brought in with `eval --import-vectors ZIP --models qwen3-4b`. It only answers the
    # evaluation questions it was given.
    "qwen3-4b": {
        "model_name": "Qwen/Qwen3-Embedding-4B", "revision": "5cf2132abc99cad020ac570b19d031efec650f2b",
        "query_prefix": QWEN_QUERY_PREFIX, "document_prefix": "", "max_seq_length": 8192,
        "precomputed": True, "pooling": "lasttoken",
    },
}
