"""Search the index by text OR by example image — one path, many inputs."""
from __future__ import annotations

import re
from pathlib import Path

from .db import get_cached_table
from .detect import COCO_CLASSES, like_clause
from .embed import embed_image, embed_text

# Map everyday query words to the COCO class YOLO actually detects, so "a woman"
# / "people" / "guy" all resolve to the "person" presence-filter.
_SINGLE_WORD_CLASSES = {c for c in COCO_CLASSES if " " not in c}
QUERY_SYNONYMS = {
    "woman": "person", "women": "person", "man": "person", "men": "person",
    "people": "person", "boy": "person", "girl": "person", "child": "person",
    "children": "person", "kid": "person", "guy": "person", "lady": "person",
    "human": "person", "puppy": "dog", "kitten": "cat", "bike": "bicycle",
    "motorbike": "motorcycle", "plane": "airplane", "television": "tv",
    "computer": "laptop", "phone": "cell phone", "cellphone": "cell phone",
    "sofa": "couch",
}


def _classes_in_query(query: str) -> set[str]:
    """Which COCO object classes does this query mention?"""
    ql = query.lower()
    found = set()
    for cls in COCO_CLASSES:                 # multi-word classes: substring match
        if " " in cls and cls in ql:
            found.add(cls)
    for word in re.findall(r"[a-z]+", ql):   # single words: exact class or synonym
        if word in _SINGLE_WORD_CLASSES:
            found.add(word)
        elif word in QUERY_SYNONYMS:
            found.add(QUERY_SYNONYMS[word])
    return found


def _run(query_vector, limit: int, classes=frozenset()):
    table = get_cached_table()  # cached: avoids re-opening the DB every search
    if table.count_rows() == 0:
        return []

    def run(where=None):
        q = table.search(query_vector.tolist()).metric("cosine")  # numpy -> list
        if where:
            # prefilter=True applies the object filter BEFORE the KNN, so we
            # still get `limit` results that actually contain the object.
            q = q.where(where, prefilter=True)
        return q.limit(limit).to_list()  # top-k rows as dicts

    where = " AND ".join(like_clause(c) for c in classes) if classes else None
    rows = run(where)
    if where and not rows:
        rows = run(None)  # graceful fallback if the filter excluded everything
    results = []
    for r in rows:
        # LanceDB returns cosine *distance*; similarity = 1 - distance.
        results.append({
            "path": r["path"],
            "score": round(1.0 - r["_distance"], 4),
        })
    return results


def search_text(query: str, limit: int = 20, require_objects: bool = True):
    """Text search. If the query names an object (a woman, a dog, pizza...),
    require that object to actually be present (per YOLO) — not just present in
    CLIP's overall "vibe" for the scene.
    """
    classes = _classes_in_query(query) if require_objects else frozenset()
    return _run(embed_text(query), limit, classes)


def search_by_image(image_path: str | Path, limit: int = 20):
    """Find images similar to an example image"""
    return _run(embed_image(image_path), limit)