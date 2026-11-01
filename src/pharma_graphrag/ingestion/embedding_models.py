"""Embedding model loading, normalized behind one interface regardless of how each
model is actually packaged on HuggingFace Hub.

Models are referenced throughout this project by their full HuggingFace repo ID (e.g.
"cambridgeltl/SapBERT-from-PubMedBERT-fulltext"), not a short alias. That ID is the
canonical, unambiguous identifier, and it's what ends up in results CSVs. A short alias
would be more convenient to type, but a results CSV is a reproducibility record: it
should say exactly which model produced a number, not a name whose mapping lives in
mutable source code that could point somewhere else by the time someone reads an old
result.

Some models ship as native sentence-transformers checkpoints (mean-pooled, ready to
use via SentenceTransformer). Others - including SapBERT and CODER, the two models
most relevant to biomedical synonym matching - ship as raw `transformers` BERT-family
checkpoints, conventionally embedded via the [CLS] token of the last hidden state.
Both are wrapped behind the same `.encode(texts) -> np.ndarray` call so the rest of
the pipeline never needs to know which loading path a given model takes.
"""

from __future__ import annotations

import numpy as np

# Full HuggingFace repo ID -> loading strategy ("sentence_transformers" or
# "transformers_cls"). The repo ID is the registry key and the identifier used
# everywhere else in the pipeline - no separate short name.
EMBEDDING_MODEL_STRATEGIES: dict[str, str] = {
    "sentence-transformers/all-MiniLM-L6-v2": "sentence_transformers",
    "sentence-transformers/all-mpnet-base-v2": "sentence_transformers",
    "FremyCompany/BioLORD-2023": "sentence_transformers",
    "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext": "transformers_cls",
    "cambridgeltl/SapBERT-from-PubMedBERT-fulltext": "transformers_cls",
    "cambridgeltl/SapBERT-UMLS-2020AB-all-lang-from-XLMR": "transformers_cls",
    "GanjinZero/coder_eng": "transformers_cls",
}


class EmbeddingModel:
    """Lazily loads one embedding model, identified by its full HuggingFace repo ID,
    and exposes a uniform encode() call."""

    def __init__(self, hf_repo_id: str, batch_size: int = 32, max_length: int = 64):
        if hf_repo_id not in EMBEDDING_MODEL_STRATEGIES:
            raise ValueError(
                f"Unknown embedding model {hf_repo_id!r}. Known: {sorted(EMBEDDING_MODEL_STRATEGIES)}"
            )
        self.hf_repo_id = hf_repo_id
        self.strategy = EMBEDDING_MODEL_STRATEGIES[hf_repo_id]
        self.batch_size = batch_size
        self.max_length = max_length
        self._st_model = None
        self._tokenizer = None
        self._hf_model = None

    def _ensure_loaded(self) -> None:
        if self.strategy == "sentence_transformers":
            if self._st_model is not None:
                return
            from sentence_transformers import SentenceTransformer

            self._st_model = SentenceTransformer(self.hf_repo_id)
        else:
            if self._hf_model is not None:
                return
            from transformers import AutoModel, AutoTokenizer

            self._tokenizer = AutoTokenizer.from_pretrained(self.hf_repo_id)
            self._hf_model = AutoModel.from_pretrained(self.hf_repo_id)
            self._hf_model.eval()

    def encode(self, texts: list[str]) -> np.ndarray:
        self._ensure_loaded()
        if self.strategy == "sentence_transformers":
            return np.asarray(self._st_model.encode(texts, normalize_embeddings=True))
        return self._encode_cls(texts)

    def _encode_cls(self, texts: list[str]) -> np.ndarray:
        import torch

        all_embeddings = []
        for start in range(0, len(texts), self.batch_size):
            batch = texts[start : start + self.batch_size]
            inputs = self._tokenizer(
                batch, padding=True, truncation=True, max_length=self.max_length, return_tensors="pt"
            )
            with torch.no_grad():
                outputs = self._hf_model(**inputs)
            cls_embeddings = outputs.last_hidden_state[:, 0, :].numpy()
            all_embeddings.append(cls_embeddings)
        embeddings = np.concatenate(all_embeddings, axis=0)
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return embeddings / norms


def load_embedding_model(hf_repo_id: str) -> EmbeddingModel:
    return EmbeddingModel(hf_repo_id)
