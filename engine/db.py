"""
I used LanceDB as vectordb because it is fully local, embeddable and can run on MacOS, Windows and Linux. 
LanceDB scales to 100k+ vectors. 
"""

from __future__ import annotations

from pathlib import Path

import lancedb
from lancedb.pydantic import LanceModel, Vector

from .embed import EMBED_DIM

DEFAULT_DB_DIR = Path.home() / ".lumen" / "index" #It is a container which can hold multiple independent named tables.
TABLE_NAME = "images" #Table where the image embeddings are stored.

class ImageRecord(LanceModel):
    """
    {
        'path': '/Users/.../Screenshot 2026-07-19 at 5.09.21 PM.png',
        'mtime': 1784461166.68,
        'size': 109525,
        'vector': [0.0044, 0.0295, -0.0192, ...]   # 512 floats total
    }
    """
    path: str  # absolute file path (also our unique id)
    mtime: float  # last-modified time, for incremental re-indexing later
    size: int
    vector: Vector(EMBED_DIM)


def connect(db_dir: str | Path = DEFAULT_DB_DIR):
    db_dir = Path(db_dir)
    db_dir.mkdir(parents=True, exist_ok=True)
    return lancedb.connect(str(db_dir))


def get_table(db_dir: str | Path = DEFAULT_DB_DIR):
    """Open the images table, creating it (empty) on first use."""
    db = connect(db_dir)
    if TABLE_NAME in db.table_names():
        return db.open_table(TABLE_NAME)
    return db.create_table(TABLE_NAME, schema=ImageRecord)
