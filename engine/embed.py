"""
CLIP (Contrastive Language-Image Pre-training, OpenAI) is a model trained to push text and images into the same vector space (512-dim), so a caption and a matching photo end up as nearby vectors.
Since PyTorch is heavy dependency, I used the ONNX version of CLIP via fastembed.
fastembed is Qdrant's library that wraps ONNX CLIP models with a simple .embed() API — it downloads/caches the ONNX weights and handles preprocessing (tokenizing text, resizing/normalizing images) for you.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Iterable

import numpy as np

MODEL_CACHE_DIR = Path.home() / ".lumen" / "models" #This is where we cache the ONNX models, so we don't have to redownload them every time.

# Matched CLIP encoders — both halves of the SAME model, so their outputs live in one comparable space.
TEXT_MODEL = "Qdrant/clip-ViT-B-32-text"
IMAGE_MODEL = "Qdrant/clip-ViT-B-32-vision"
EMBED_DIM = 512

@lru_cache(maxsize=1)
def _text_model():
    from fastembed import TextEmbedding

    return TextEmbedding(model_name=TEXT_MODEL, cache_dir=MODEL_CACHE_DIR)


@lru_cache(maxsize=1)
def _image_model():
    from fastembed import ImageEmbedding

    return ImageEmbedding(model_name=IMAGE_MODEL, cache_dir=MODEL_CACHE_DIR)


def _normalize(vec: np.ndarray) -> np.ndarray:
    """L2-normalize (Euclidean norm) so magnitude of vector becomes 1, making the dot product == cosine similarity."""
    norm = np.linalg.norm(vec)
    return vec / norm if norm else vec


def embed_text(text: str) -> np.ndarray:
    """
    This is the public function other code calls to turn a search query string into a vector.
    """
    vec = next(_text_model().embed([text])) #fastembed returns a generator, so we have to call next() to get the first (and only) vector.
    return _normalize(np.asarray(vec, dtype=np.float32))


def embed_image(path: str | Path) -> np.ndarray:
    """
    Same as embed_text(), but for a single image path. Returns a normalized vector.
    """
    vec = next(_image_model().embed([str(path)]))
    return _normalize(np.asarray(vec, dtype=np.float32))


def embed_images(paths: Iterable[str | Path]):
    """Batch-embed image paths, yielding (path, normalized_vector). Default batch size = 16 for images and 256 for text.

    Batching keeps ONNX inference fast at large scale. Paths that fail to
    load (corrupt / unreadable) are skipped rather than aborting the run.
    """
    paths = [str(p) for p in paths]
    for path, vec in zip(paths, _image_model().embed(paths)):
        yield path, _normalize(np.asarray(vec, dtype=np.float32))
