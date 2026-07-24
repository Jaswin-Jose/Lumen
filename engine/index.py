"""Crawl folders for images and build the searchable vector index."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Iterator

from .db import get_table
from .embed import embed_images

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp", ".tiff", ".heic"}

# Folders we never want to crawl when indexing a whole system.
SKIP_DIRS = {
    "node_modules", ".git", "__pycache__", ".cache", "Library",
    "site-packages", ".Trash", "venv", ".venv",
}


def find_images(root: str | Path) -> Iterator[Path]:
    """Yield image files under root, skipping noise directories."""
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS and not d.startswith(".")] # Slice assignment mutates dirnames in place (same object os.walk holds areference to), so pruning here actually skips these dirs during the walk plain reassignment wouldn't, since os.walk would still see the old list.
        for name in filenames:
            if Path(name).suffix.lower() in IMAGE_EXTS:
                yield Path(dirpath) / name


def index_folder(root: str | Path, batch_size: int = 32, progress=None) -> int:
    """Index all images under root. Returns the number of images added.
    progress is discussed in cli.py, and is a callable that takes (done, path) and prints progress to the user.
    """
    table = get_table()

    # Skip files already indexed and unchanged.
    known: dict[str, float] = {}
    if table.count_rows() > 0:
        for row in table.search().select(["path", "mtime"]).limit(10_000_000).to_list(): #The lance by default returns 10 entries, so we put a large limit to get all the entries. Though, its not good for large datasets as it causes memory issues, it may be fine because we take only row and mtime. 
            known[row["path"]] = row["mtime"]

    todo: list[Path] = []
    for path in find_images(root):
        try:
            st = path.stat() #returns mtime, size, ctime (metadata-change time), atime (last-access time)...
        except OSError:
            continue
        if known.get(str(path)) == st.st_mtime: #if both mtime stored in the dictionary and the current mtime are same, skip it. Else add it to todo list for indexing.
            continue
        todo.append(path)

    added = 0
    batch: list[dict] = []
    for done, (path_str, vector) in enumerate(embed_images(todo), start=1):
        try:
            st = os.stat(path_str) #path_str is a plain string, and strings have no .stat(). os.stat(path_str) is the more direct route — string goes straight to the syscall, nothing built in between. 
        except OSError:
            continue
        batch.append({
            "path": path_str,
            "mtime": st.st_mtime,
            "size": st.st_size,
            "vector": vector.tolist(),
        })
        if len(batch) >= batch_size:
            table.add(batch)
            added += len(batch)
            batch = []

        if progress:
            progress(done, path_str)

    if batch: #to catch any remaining images that didn't fill a complete batch.
        table.add(batch)
        added += len(batch)
    return added