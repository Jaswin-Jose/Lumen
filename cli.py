#!/usr/bin/env python3
"""Lumen CLI — prove the engine.
    python cli.py index  ~/Pictures
    python cli.py search "a blue fish"
    python cli.py similar ~/Pictures/reef.jpg
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime


def cmd_index(args):
    from engine.index import index_folder

    start = time.time()

    def progress(done, path):
        if done % 10 == 0:
            print(f"  ...{done} images", end="\r", flush=True) #for live updating counter

    print(f"Indexing {args.folder} (first run downloads the CLIP model)...")
    res = index_folder(args.folder, progress=progress)
    print(f"\nIndexed {res.added} new image(s) in {time.time() - start:.1f}s")
    if res.skipped or res.failed:
        print(f"Skipped {res.skipped} (vanished/unreadable), "
              f"{res.failed} failed to decode. See {res.ledger_path}")


def cmd_reset(args):
    from engine.db import drop_table

    print("Index dropped." if drop_table() else "No index to drop.")


def cmd_prune(args):
    from engine.index import prune_missing

    n = prune_missing()
    print(f"Pruned {n} entr{'y' if n == 1 else 'ies'} for files no longer on disk.")


def cmd_compact(args):
    from engine.db import compact

    before, after, stats = compact(retention_days=args.days)
    freed = ""
    try:  # stats shape varies across lancedb versions; surface bytes if present
        freed = f", freed {stats.prune.bytes_removed / 1e6:.1f} MB"
    except AttributeError:
        pass
    print(f"Compacted: {before} -> {after} versions{freed}")


def cmd_failures(args):
    from engine.index import LEDGER_PATH

    if not LEDGER_PATH.exists():
        print("No failure ledger yet — nothing has failed to index.")
        return
    lines = LEDGER_PATH.read_text().splitlines()
    if not lines:
        print("Failure ledger is empty.")
        return

    tail = lines[-args.limit:] if args.limit > 0 else lines
    print(f"Showing last {len(tail)} of {len(lines)} failure(s) — {LEDGER_PATH}")
    for line in tail:
        try:
            r = json.loads(line)
            ts = datetime.fromtimestamp(r.get("ts", 0)).strftime("%Y-%m-%d %H:%M")
            print(f"  [{ts}] {r.get('stage', '?'):5}  {r.get('path', '?')}  — {r.get('error', '')}")
        except ValueError:  # not valid JSON — show the raw line
            print(f"  {line}")


def _show(results):
    if not results:
        print("No matches. Have you indexed a folder yet?")
        return
    for i, r in enumerate(results, 1):
        print(f"{i:>2}. {r['score']:.3f}  {r['path']}")


def cmd_search(args):
    from engine.search import search_text

    _show(search_text(args.query, limit=args.limit,
                      require_objects=not args.no_objects))


def cmd_similar(args):
    from engine.search import search_by_image

    _show(search_by_image(args.image, limit=args.limit))


def main(argv=None):
    p = argparse.ArgumentParser(prog="Lumen")
    sub = p.add_subparsers(dest="cmd", required=True)

    pi = sub.add_parser("index", help="index images under a folder")
    pi.add_argument("folder")
    pi.set_defaults(func=cmd_index)

    ps = sub.add_parser("search", help="search by text")
    ps.add_argument("query")
    ps.add_argument("--no-objects", action="store_true",
                    help="disable YOLO object-presence filtering (pure CLIP)")
    ps.add_argument("-n", "--limit", type=int, default=20)
    ps.set_defaults(func=cmd_search)

    pm = sub.add_parser("similar", help="search by example image")
    pm.add_argument("image")
    pm.add_argument("-n", "--limit", type=int, default=20)
    pm.set_defaults(func=cmd_similar)

    pr = sub.add_parser("reset", help="drop the whole index (rebuild from scratch)")
    pr.set_defaults(func=cmd_reset)

    pp = sub.add_parser("prune", help="remove entries for files no longer on disk")
    pp.set_defaults(func=cmd_prune)

    pc = sub.add_parser("compact", help="reclaim disk: compact + prune old index versions")
    pc.add_argument("--days", type=float, default=7.0,
                    help="keep versions newer than this many days (default 7)")
    pc.set_defaults(func=cmd_compact)

    pfl = sub.add_parser("failures", help="show recent entries from the failure ledger")
    pfl.add_argument("-n", "--limit", type=int, default=20,
                     help="how many lines from the end (0 = all, default 20)")
    pfl.set_defaults(func=cmd_failures)

    args = p.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    sys.exit(main())
