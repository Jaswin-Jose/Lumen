"""
I used LanceDB as vectordb because it is fully local, embeddable and can run on MacOS, Windows and Linux. 
LanceDB scales to 100k+ vectors. 
"""

from __future__ import annotations

from datetime import timedelta
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
    labels: str  # detected objects, delimited: "|person|chair|tv|" (see detect.py)
    vector: Vector(EMBED_DIM)


def connect(db_dir: str | Path = DEFAULT_DB_DIR):
    db_dir = Path(db_dir)
    db_dir.mkdir(parents=True, exist_ok=True)
    return lancedb.connect(str(db_dir))


def get_table(db_dir: str | Path = DEFAULT_DB_DIR):
    """Open the images table, creating it (empty) on first use.
    This always re-opens from disk. Writers (indexing) use it directly so
    they see the freshest state; the search/read path uses get_cached_table().
    """
    db = connect(db_dir)
    if TABLE_NAME in db.table_names():
        return db.open_table(TABLE_NAME)
    return db.create_table(TABLE_NAME, schema=ImageRecord)

"""
While using this as an app that answers many searches, reconnecting + re-reading the
LanceDB manifest on every query is wasted work (the same reason we cache the
CLIP models). So we cache the opened table. 
So when the indexer adds new content, we need to invalidate the cache so the next search sees the new data.
"""
_table_cache: dict[str, object] = {}


def get_cached_table(db_dir: str | Path = DEFAULT_DB_DIR):
    key = str(db_dir)
    if key not in _table_cache:
        _table_cache[key] = get_table(db_dir)
    return _table_cache[key]


def invalidate_table_cache() -> None:
    """Drop cached handles so the next read re-opens at the latest version."""
    _table_cache.clear()


def drop_table(db_dir: str | Path = DEFAULT_DB_DIR) -> bool:
    """Delete the whole images table — a true 'reset from scratch'.

    A LanceDB table is a versioned DIRECTORY tree, not a single file, and it's
    append-only (deletes just add a new version). So you can't reset it by
    removing one file; the reliable resets are `rm -rf` the index dir or, as
    here, dropping the table. Returns True if a table existed.
    """
    db = connect(db_dir)
    existed = TABLE_NAME in db.table_names()
    if existed:
        db.drop_table(TABLE_NAME)
    invalidate_table_cache()
    return existed


def compact(db_dir: str | Path = DEFAULT_DB_DIR, retention_days: float = 7.0):
    """
    LanceDB never deletes on its own — every index/prune/delete leaves a new
    version behind, so for an app meant to run indefinitely, disk quietly grows
    forever unless this runs. optimize() does the two maintenance jobs at once:
      * compacts many small data files from incremental adds into fewer big
        ones (keeps reads fast), and
      * drops versions OLDER than the retention window (recent history is kept
        so you can still recover / time-travel; anything older is reclaimed).
    Deliberate and manual by design — call it on a schedule, not per index.
    Returns (versions_before, versions_after, stats).
    """
    table = get_table(db_dir)
    before = len(table.list_versions())
    stats = table.optimize(cleanup_older_than=timedelta(days=retention_days))
    after = len(table.list_versions())
    invalidate_table_cache()
    return before, after, stats