"""Embedding model wrapper, so the model can be swapped without touching the rest."""

import numpy as np

from patent_search import config


class Embedder:
    def __init__(self, model_name: str = config.MODEL_NAME,
                 revision: str = config.MODEL_REVISION,
                 code_revision: str | None = config.MODEL_CODE_REVISION,
                 document_prefix: str = config.DOCUMENT_PREFIX,
                 query_prefix: str = config.QUERY_PREFIX,
                 batch_size: int = config.EMBED_BATCH_SIZE,
                 trust_remote_code: bool = True,
                 device: str | None = None):
        from sentence_transformers import SentenceTransformer

        self.model_name = model_name
        self.revision = revision
        self.code_revision = code_revision
        self.document_prefix = document_prefix
        self.query_prefix = query_prefix
        self.batch_size = batch_size
        pinned = {"code_revision": code_revision} if code_revision else {}
        self.model = SentenceTransformer(
            model_name, revision=revision, trust_remote_code=trust_remote_code,
            model_kwargs=pinned, config_kwargs=pinned, device=device,
        )

    @property
    def dimension(self) -> int:
        # get_embedding_dimension is the newer name; older versions only have the long one.
        get = getattr(self.model, "get_embedding_dimension", None) or self.model.get_sentence_embedding_dimension
        return get()

    def _encode(self, texts: list[str], show_progress: bool) -> np.ndarray:
        vectors = self.model.encode(
            texts, batch_size=self.batch_size, normalize_embeddings=True,
            convert_to_numpy=True, show_progress_bar=show_progress,
        )
        return vectors.astype(np.float32)

    def embed_documents(self, texts: list[str], show_progress: bool = False) -> np.ndarray:
        return self._encode([self.document_prefix + t for t in texts], show_progress)

    def embed_query(self, text: str) -> np.ndarray:
        return self._encode([self.query_prefix + text], False)[0]
