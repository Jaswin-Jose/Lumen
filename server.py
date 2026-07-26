"""Local HTTP sidecar for the Lumen desktop app.

Electron spawns this once and talks to it over 127.0.0.1. The ONNX models (jina + YOLO) and the
cached LanceDB handle stay warm, so every search after the first is ~15 ms
instead of paying the ~430 ms cold-start each time.

API is under /api/*; the Electron renderer (static files) is served at / so the
whole app is one origin (no CORS).
"""
from __future__ import annotations

import argparse
import io
import threading
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from PIL import Image

from engine import embed
from engine.db import get_cached_table
from engine.index import index_folder
from engine.search import search_by_image, search_text

APP_DIR = Path(__file__).parent / "app" / "renderer"

app = FastAPI(title="Lumen")

# ---- background indexing state (one job at a time) ----
_index_state = {
    "running": False, "folder": None, "done": 0,
    "added": 0, "skipped": 0, "failed": 0, "error": None,
}
_index_lock = threading.Lock()
"""
1. You click "index this folder" in the app.
The Electron window sends one HTTP request: POST /api/index?folder=....

2. Thread A picks up that request and runs start_index.
It locks _index_state, checks "is a job already running?" — no — so it writes running=True, resets the counters to 0, then unlocks. Thread A then creates Thread B (this is the moment the worker thread is born) and tells it to start running _run_index. Thread A immediately replies {"started": true} to the app and is now done — it goes back to being available for other requests.

3. Thread B starts the real work.
It calls index_folder(folder), which starts crawling the folder and embedding images, one at a time — this is the slow part, could take minutes.

4. Every single time Thread B finishes one image, it briefly locks _index_state, writes the new done count (e.g. 1, then 2, then 3...), and unlocks immediately. It's locked for a tiny instant each time, then goes right back to working on the next image, unlocked.

5. Meanwhile, the app is repeatedly asking "how's it going?" — roughly once a second, it sends GET /api/index/status. Each of these requests gets picked up by some Thread C (could be a different one each time, doesn't matter which). Thread C locks _index_state, copies whatever's in there right now (e.g. done: 250), unlocks, and sends that back to the app so the progress bar can update. This keeps happening, over and over, the entire time Thread B is working.

6. Thread B and the various Thread Cs are running at the same time, constantly, for the whole duration of the job. That's the whole reason for the lock: Thread B is writing done very frequently, and Thread Cs are reading it very frequently, completely independently of each other — the lock just makes sure a read never happens in the middle of a write (which could show garbage/half-updated data).

7. Eventually, Thread B finishes all the images.
It locks one last time, writes the final added/skipped/failed numbers, and sets running=False. Then Thread B's job is done — it ends.

8. The next time a Thread C polls status, it sees running=False and the final numbers. The app shows "done," stops polling.

"""
def _fmt(results): 
    "Format each result as a dict with path, name, and score. "
    return [
        {"path": r["path"], "name": Path(r["path"]).name, "score": r["score"]}
        for r in results
    ]


@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.get("/api/stats")
def stats():
    try:
        n = get_cached_table().count_rows()
    except Exception:
        n = 0
    return {"count": n}


@app.get("/api/search")
def search(q: str, limit: int = 60, objects: bool = True):
    return {"results": _fmt(search_text(q, limit=limit, require_objects=objects))}
# run this to test: .venv/bin/python server.py --port 8765 && curl "http://127.0.0.1:8765/api/search?q=a%20blue%20fish&limit=10"

@app.get("/api/similar")
def similar(path: str, limit: int = 60):
    try:
        return {"results": _fmt(search_by_image(path, limit=limit))}
    except Exception as e:
        return JSONResponse({"error": repr(e)}, status_code=400)


@app.get("/api/thumb")
def thumb(path: str, size: int = 320):
    """Serve a resized JPEG so the grid stays light even at 100k images."""
    try:
        with Image.open(path) as im:
            im = im.convert("RGB")
            im.thumbnail((size, size))
            buf = io.BytesIO()
            im.save(buf, "JPEG", quality=82)
    except Exception:
        return Response(status_code=404)
    return Response(buf.getvalue(), media_type="image/jpeg")


def _run_index(folder: str):
    def progress(done, _path):
        with _index_lock:
            """
            with with, the lock correctly released itself even though 
            an exception happened mid-block. Without it, using plain 
            .acquire()/.release(), the crash skipped right past .release()
            and left the lock permanently stuck locked — every future request
            needing that lock would hang forever, since nothing would 
            ever release it.
            """
            _index_state["done"] = done
    try:
        res = index_folder(folder, progress=progress) #we pass the progress function to index_folder, which will call it every time an image is processed. The progress function updates the _index_state with the number of images processed so far.
        with _index_lock:
            _index_state.update(added=res.added, skipped=res.skipped, failed=res.failed)
    except Exception as e:  # never let the worker thread die silently
        with _index_lock:
            _index_state["error"] = repr(e)
    finally:
        with _index_lock:
            _index_state["running"] = False


@app.post("/api/index")
def start_index(folder: str):
    with _index_lock:
        if _index_state["running"]:
            return JSONResponse({"error": "indexing already running"}, status_code=409)
        _index_state.update(running=True, folder=folder, done=0,
                            added=0, skipped=0, failed=0, error=None)
    threading.Thread(target=_run_index, args=(folder,), daemon=True).start() #worker thread created as daemon so it wont block main thread from exiting.
    return {"started": True}


@app.get("/api/index/status")
def index_status():
    with _index_lock:
        return dict(_index_state)


# Static renderer at / (registered last so it doesn't shadow /api/*).
if APP_DIR.exists():
    app.mount("/", StaticFiles(directory=str(APP_DIR), html=True), name="app")


def _warmup():
    """Load the text/image towers in the background so the first search is fast."""
    try:
        embed.embed_text("warmup")
    except Exception:
        pass


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--host", default="127.0.0.1")
    args = ap.parse_args()

    threading.Thread(target=_warmup, daemon=True).start()
    import uvicorn
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")
