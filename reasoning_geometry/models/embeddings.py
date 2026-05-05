from __future__ import annotations

from functools import lru_cache
from typing import Dict, Iterable, List

import numpy as np
import torch
from sklearn.feature_extraction.text import TfidfVectorizer

from reasoning_geometry.common import cosine_similarity_matrix


def surface_distance_from_embeddings(embeddings: np.ndarray) -> np.ndarray:
    similarity = cosine_similarity_matrix(embeddings)
    return (1.0 - similarity).astype(np.float32)


def tfidf_embeddings(step_texts: List[str]) -> np.ndarray:
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1)
    matrix = vectorizer.fit_transform(step_texts)
    return matrix.toarray().astype(np.float32)


@lru_cache(maxsize=1)
def _load_sbert():
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise ImportError(
            "sentence-transformers is required for SBERT surface embeddings."
        ) from exc
    return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")


def sbert_embeddings(step_texts: List[str]) -> np.ndarray:
    model = _load_sbert()
    embeddings = model.encode(step_texts, convert_to_numpy=True, normalize_embeddings=True)
    return embeddings.astype(np.float32)


def model_first_token_embeddings(
    step_texts: List[str],
    tokenizer,
    model,
    device: str,
    dtype: torch.dtype,
) -> np.ndarray:
    outputs: List[np.ndarray] = []
    with torch.no_grad():
        for text in step_texts:
            encoded = tokenizer(text, return_tensors="pt", truncation=True)
            encoded = {key: value.to(device) for key, value in encoded.items()}
            input_ids = encoded["input_ids"]
            embeddings = model.get_input_embeddings()(input_ids).to(dtype=torch.float32)
            outputs.append(embeddings[0, 0].detach().cpu().numpy())
    return np.stack(outputs).astype(np.float32)


def compute_surface_distances(
    step_texts: List[str],
    tokenizer=None,
    model=None,
    device: str = "cpu",
    dtype: torch.dtype = torch.float32,
    modes: Iterable[str] = ("tfidf", "sbert", "model_first_token"),
) -> Dict[str, np.ndarray]:
    result: Dict[str, np.ndarray] = {}
    for mode in modes:
        if mode == "tfidf":
            embeddings = tfidf_embeddings(step_texts)
        elif mode == "sbert":
            embeddings = sbert_embeddings(step_texts)
        elif mode == "model_first_token":
            if tokenizer is None or model is None:
                raise ValueError("Tokenizer/model required for model_first_token embeddings.")
            embeddings = model_first_token_embeddings(step_texts, tokenizer, model, device, dtype)
        else:
            raise ValueError(f"Unknown surface embedding mode: {mode}")
        result[mode] = surface_distance_from_embeddings(embeddings)
    return result
