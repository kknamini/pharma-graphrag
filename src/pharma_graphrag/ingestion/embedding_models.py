"""Embedding model loading, normalized behind one interface regardless of how each
model is actually packaged on HuggingFace Hub.

Some models ship as native sentence-transformers checkpoints (mean-pooled, ready to
use via SentenceTransformer). Others - including SapBERT and CODER, the two models
most relevant to biomedical synonym matching - ship as raw `transformers` BERT-family
checkpoints, conventionally embedded via the [CLS] token of the last hidden state.
Both are wrapped behind the same `.encode(texts) -> np.ndarray` call so the rest of
the pipeline never needs to know which loading path a given model takes.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class EmbeddingModelSpec:
    name: str
    hf_repo_id: str
    strategy: str  # "sentence_transformers" | "transformers_cls"


EMBEDDING_MODELS: dict[str, EmbeddingModelSpec] = {
    "minilm": EmbeddingModelSpec(
        "minilm", "sentence-transformers/all-MiniLM-L6-v2", "sentence_transformers"
    ),
    "mpnet": EmbeddingModelSpec(
        "mpnet", "sentence-transformers/all-mpnet-base-v2", "sentence_transformers"
    ),
    "biolord": EmbeddingModelSpec(
        "biolord", "FremyCompany/BioLORD-2023", "sentence_transformers"
    ),
    "pubmedbert_base": EmbeddingModelSpec(
        "pubmedbert_base",
        "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext",
        "transformers_cls",
    ),
    "sapbert": EmbeddingModelSpec(
        "sapbert", "cambridgeltl/SapBERT-from-PubMedBERT-fulltext", "transformers_cls"
    ),
    "sapbert_multilingual": EmbeddingModelSpec(
        "sapbert_multilingual",
        "cambridgeltl/SapBERT-UMLS-2020AB-all-lang-from-XLMR",
        "transformers_cls",
    ),
    "coder": EmbeddingModelSpec("coder", "GanjinZero/coder_eng", "transformers_cls"),
}


class EmbeddingModel:
    """Lazily loads one embedding model and exposes a uniform encode() call."""

    def __init__(self, spec: EmbeddingModelSpec, batch_size: int = 32, max_length: int = 64):
        self.spec = spec
        self.batch_size = batch_size
        self.max_length = max_length
        self._st_model = None
        self._tokenizer = None
        self._hf_model = None

    def _ensure_loaded(self) -> None:
        if self.spec.strategy == "sentence_transformers":
            if self._st_model is not None:
                return
            from sentence_transformers import SentenceTransformer

            self._st_model = SentenceTransformer(self.spec.hf_repo_id)
        else:
            if self._hf_model is not None:
                return
            from transformers import AutoModel, AutoTokenizer

            self._tokenizer = AutoTokenizer.from_pretrained(self.spec.hf_repo_id)
            self._hf_model = AutoModel.from_pretrained(self.spec.hf_repo_id)
            self._hf_model.eval()

    def encode(self, texts: list[str]) -> np.ndarray:
        self._ensure_loaded()
        if self.spec.strategy == "sentence_transformers":
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


def load_embedding_model(name: str) -> EmbeddingModel:
    if name not in EMBEDDING_MODELS:
        raise ValueError(f"Unknown embedding model {name!r}. Known: {sorted(EMBEDDING_MODELS)}")
    return EmbeddingModel(EMBEDDING_MODELS[name])
