#!/usr/bin/env python3
"""Lumen CLI — prove the engine.
    python cli.py index  ~/Pictures
    python cli.py search "a blue fish"
    python cli.py similar ~/Pictures/reef.jpg
"""
from __future__ import annotations

import argparse
import sys
import time


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


def _show(results):
    if not results:
        print("No matches. Have you indexed a folder yet?")
        return
    for i, r in enumerate(results, 1):
        print(f"{i:>2}. {r['score']:.3f}  {r['path']}")


def cmd_search(args):
    from engine.search import search_text

    _show(search_text(args.query, limit=args.limit))


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

    args = p.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    sys.exit(main())
