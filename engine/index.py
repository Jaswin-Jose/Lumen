"""Crawl folders for images and build the searchable vector index."""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

from .db import DEFAULT_DB_DIR, get_table, invalidate_table_cache
from .embed import embed_images
"""
One JSONL line per file we couldn't index, with the stage that failed and why.
"""
LEDGER_PATH = DEFAULT_DB_DIR.parent / "failed.jsonl"

# The ledger is append-only, so on a machine running for years it would grow
# without bound. Cap it: once the file crosses MAX_LEDGER_BYTES, keep only the
# most recent MAX_LEDGER_LINES entries (newest failures are the useful ones).
MAX_LEDGER_BYTES = 5 * 1024 * 1024  # 5 MB
MAX_LEDGER_LINES = 5_000


@dataclass
class IndexResult:
    added: int = 0        # embeddings written to the table
    skipped: int = 0      # files that vanished / became unreadable mid-run
    failed: int = 0       # images fastembed/PIL couldn't decode
    ledger: list = field(default_factory=list)  # detail rows for the two above

    @property
    def ledger_path(self) -> str | None:
        return str(LEDGER_PATH) if self.ledger else None

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


def index_folder(root: str | Path, batch_size: int = 32, progress=None) -> IndexResult:
    """Index all images under root. Returns an IndexResult (added/skipped/failed).
    progress is discussed in cli.py, and is a callable that takes (done, path) and prints progress to the user.
    """
    table = get_table()  # writer path: always fresh, so we see our own writes
    result = IndexResult()

    # Skip files already indexed and unchanged.
    known: dict[str, float] = {}
    if table.count_rows() > 0:
        for row in table.search().select(["path", "mtime"]).limit(10_000_000).to_list(): #The lance by default returns 10 entries, so we put a large limit to get all the entries. Though, its not good for large datasets as it causes memory issues, it may be fine because we take only row and mtime.
            known[row["path"]] = row["mtime"]

    todo: list[Path] = []
    for path in find_images(root):
        try:
            st = path.stat() #returns mtime, size, ctime (metadata-change time), atime (last-access time)...
        except OSError as e:
            # Race: file listed by os.walk but gone/unreadable by stat(). Don't
            # crash and don't skip silently — record it so it leaves a trace.
            result.skipped += 1
            result.ledger.append({"path": str(path), "stage": "stat", "error": repr(e)})
            continue
        if known.get(str(path)) == st.st_mtime: #if both mtime stored in the dictionary and the current mtime are same, skip it. Else add it to todo list for indexing.
            continue
        todo.append(path)

    # embed_images appends corrupt/undecodable images here instead of crashing.
    embed_failed: list = []
    batch: list[dict] = []
    for done, (path_str, vector) in enumerate(embed_images(todo, failed=embed_failed), start=1):
        try:
            st = os.stat(path_str) #path_str is a plain string, and strings have no .stat(). os.stat(path_str) is the more direct route — string goes straight to the syscall, nothing built in between.
        except OSError as e:
            result.skipped += 1
            result.ledger.append({"path": path_str, "stage": "stat", "error": repr(e)})
            continue
        batch.append({
            "path": path_str,
            "mtime": st.st_mtime,
            "size": st.st_size,
            "vector": vector.tolist(),
        })
        if len(batch) >= batch_size:
            table.add(batch)
            result.added += len(batch)
            batch = []

        if progress:
            progress(done, path_str)

    if batch: #to catch any remaining images that didn't fill a complete batch.
        table.add(batch)
        result.added += len(batch)

    for f in embed_failed:
        result.failed += 1
        result.ledger.append({"path": f["path"], "stage": "embed", "error": f["error"]})

    invalidate_table_cache()  # searches must re-open to see what we just wrote
    _write_ledger(result.ledger)
    return result


def _write_ledger(rows: list) -> None:
    """Append failure/skip rows to ~/.lumen/failed.jsonl (one JSON per line)."""
    if not rows:
        return
    LEDGER_PATH.parent.mkdir(parents=True, exist_ok=True)
    ts = time.time()
    with open(LEDGER_PATH, "a") as fh:
        for row in rows:
            fh.write(json.dumps({**row, "ts": ts}) + "\n")
    _rotate_ledger()


def _rotate_ledger() -> None:
    """Trim the ledger to its most recent lines once it grows past the cap.

    The size check is cheap and almost always the only thing that runs; the
    rewrite happens only on the rare occasion the file actually gets large.
    """
    try:
        if LEDGER_PATH.stat().st_size <= MAX_LEDGER_BYTES:
            return
    except OSError:
        return
    with open(LEDGER_PATH) as fh:
        lines = fh.readlines()
    if len(lines) <= MAX_LEDGER_LINES:
        return
    with open(LEDGER_PATH, "w") as fh:
        fh.writelines(lines[-MAX_LEDGER_LINES:])


def prune_missing(progress=None) -> int:
    """
    Remove index entries whose files no longer exist on disk.
    Or else those deleted after indexing would stay forever as dead search hits.
    Deletes are done via SQL over LanceDB.
    """
    table = get_table()
    if table.count_rows() == 0:
        return 0

    missing: list[str] = []
    for row in table.search().select(["path"]).limit(10_000_000).to_list():
        if not os.path.exists(row["path"]):
            missing.append(row["path"])
        if progress:
            progress(len(missing), row["path"])

    for i in range(0, len(missing), 256):
        chunk = missing[i:i + 256]
        in_list = ",".join("'" + p.replace("'", "''") + "'" for p in chunk) #escapes any single quotes inside the path itself by doubling them.
        table.delete(f"path IN ({in_list})")
        """
        for paths ["/a.jpg", "/Jaswin's Photos/b.jpg"], this produces:
            '/a.jpg','/Jaswin''s Photos/b.jpg'
            and the final call becomes:
                table.delete("path IN ('/a.jpg','/Jaswin''s Photos/b.jpg')")
        """
    invalidate_table_cache()
    return len(missing)