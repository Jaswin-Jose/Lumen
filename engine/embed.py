"""
CLIP (Contrastive Language-Image Pre-training, OpenAI) is a model trained to push text and images into the same vector space (512-dim), so a caption and a matching photo end up as nearby vectors.
But since CLIP is less accurate with complex queries, I use jina-clip-v1. Though I may include CLIP for weaker machines in the future.
Since PyTorch is heavy dependency, I used the ONNX version of CLIP via fastembed.
fastembed is Qdrant's library that wraps ONNX CLIP models with a simple .embed() API — it downloads/caches the ONNX weights and handles preprocessing (tokenizing text, resizing/normalizing images) for you.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Iterable

import numpy as np

MODEL_CACHE_DIR = Path.home() / ".lumen" / "models" #This is where we cache the ONNX models, so we don't have to redownload them every time.

# jina-clip-v1: one model whose text and image towers share a 768-dim space.
# Stronger at descriptive/compositional retrieval than CLIP ViT-B/32, still ONNX
# on CPU (no PyTorch). 
TEXT_MODEL = "jinaai/jina-clip-v1"
IMAGE_MODEL = "jinaai/jina-clip-v1"
EMBED_DIM = 768

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


def embed_images(paths: Iterable[str | Path], failed: list | None = None,
                 chunk_size: int = 32):
    """Batch-embed image paths, yielding (path, normalized_vector).

    Fastembed opens each image with PIL inside its generator, so a
    single bad file throws mid-batch and would otherwise kill everything after
    it. To solve it, we embed in chunks and force each chunk to completion
    (list(...)) BEFORE yielding — so a mid-chunk error emits no partial results
    we'd then re-emit. If a chunk throws, we retry it one-by-one to isolate the
    culprit; files that still fail are appended to `failed` as
    {"path", "error"} and skipped instead of crashing the indexer.
    """
    paths = [str(p) for p in paths]
    for i in range(0, len(paths), chunk_size):
        chunk = paths[i:i + chunk_size]
        try:
            results = list(zip(chunk, _image_model().embed(chunk)))
        except Exception:
            """
            One bad file somewhere in the chunk — fall back to per-image so
            the good ones still get indexed and the bad one is recorded.
            """
            for p in chunk:
                try:
                    yield p, embed_image(p)
                except Exception as e:  # genuinely corrupt / unreadable
                    if failed is not None:
                        failed.append({"path": p, "error": repr(e)})
            continue
        for path, vec in results:
            yield path, _normalize(np.asarray(vec, dtype=np.float32))
