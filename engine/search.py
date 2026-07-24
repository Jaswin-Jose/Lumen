"""Search the index by text OR by example image — one path, many inputs."""
from __future__ import annotations

from pathlib import Path

from .db import get_table
from .embed import embed_image, embed_text


def _run(query_vector, limit: int):
    table = get_table()
    if table.count_rows() == 0:
        return []
    rows = (
        table.search(query_vector.tolist()) #query_vector is a numpy array, and we convert it to a list so that LanceDB can handle it. LanceDB expects a list of floats for the vector search.
        .metric("cosine")
        .limit(limit) #top k
        .to_list() #run the query, get back a list of matching rows as dicts.
    )
    results = []
    for r in rows:
        # LanceDB returns cosine *distance*; similarity = 1 - distance.
        results.append({
            "path": r["path"],
            "score": round(1.0 - r["_distance"], 4),
        })
    return results


def search_text(query: str, limit: int = 20):
    """e.g. search_text("a blue fish")"""
    return _run(embed_text(query), limit)


def search_by_image(image_path: str | Path, limit: int = 20):
    """Find images similar to an example image (drag-a-photo search)."""
    return _run(embed_image(image_path), limit)