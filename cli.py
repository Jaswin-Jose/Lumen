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
    added = index_folder(args.folder, progress=progress)
    print(f"\nIndexed {added} new image(s) in {time.time() - start:.1f}s")


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

    args = p.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    sys.exit(main())
