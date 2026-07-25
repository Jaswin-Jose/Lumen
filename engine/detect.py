"""
Object detection (YOLOv8n, ONNX, CPU) — tag each image with the COCO object
classes it contains, so a search can *require* an object to be present.

CLIP matches the overall scene. YOLO gives us a complementary signal to 
filter CLIP's results on. Runs on CPU via onnxruntime.
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np
from PIL import Image

from .embed import MODEL_CACHE_DIR

MODEL_REPO = "Shad0ws/yolov8onnx" 
MODEL_FILE = "yolov8n.onnx"
INPUT_SIZE = 640 #The fixed pixel dimensions (640×640) 
CONF_THRESHOLD = 0.35 #The confidence threshold for object detection

# The 80 COCO classes, in the order the model outputs them.
COCO_CLASSES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
    "truck", "boat", "traffic light", "fire hydrant", "stop sign",
    "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag",
    "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite",
    "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
    "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana",
    "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza",
    "donut", "cake", "chair", "couch", "potted plant", "bed", "dining table",
    "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
    "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock",
    "vase", "scissors", "teddy bear", "hair drier", "toothbrush",
]


@lru_cache(maxsize=1)
def _session():
    from huggingface_hub import hf_hub_download
    import onnxruntime as ort

    path = hf_hub_download(MODEL_REPO, MODEL_FILE, cache_dir=str(MODEL_CACHE_DIR)) #the hf_hub_download checks the given cache_dir first before downloading the model from the Hugging Face Hub. 
    return ort.InferenceSession(path, providers=["CPUExecutionProvider"]) # takes the local file path to the .onnx model file returned by hf_hub_download and loads it


def _preprocess(img: Image.Image) -> np.ndarray:
    """Letterbox to 640x640 (preserve aspect ratio) and CHW-normalize to [0,1]."""
    img = img.convert("RGB")
    w0, h0 = img.size
    scale = INPUT_SIZE / max(w0, h0)
    resized = img.resize((round(w0 * scale), round(h0 * scale))) #resizes to 640*x or x*640 (whichever is smaller is x).
    canvas = Image.new("RGB", (INPUT_SIZE, INPUT_SIZE), (114, 114, 114)) #This converts the x in previous line to 640 by adding a grey tint (114,114,114). The YONO Model trained on 114.
    canvas.paste(resized, (0, 0)) #placing the image on the canvas
    arr = np.asarray(canvas, dtype=np.float32) / 255.0 #normalize the values to 0,1 instead of 0,255
    return arr.transpose(2, 0, 1)[None]  # [1, 3, 640, 640]


def detect_labels(path, conf: float = CONF_THRESHOLD) -> list[str]:
    """Return the sorted set of COCO class names present in the image."""
    with Image.open(path) as img:
        arr = _preprocess(img) #preprocessed image.
    sess = _session()
    out = sess.run(None, {sess.get_inputs()[0].name: arr})[0][0]  # [84, 8400]
    #The syntax is sess.run(output_names, input_feed). We give None as it means -"I don't want to name specific outputs — just give me all of them." 
    # Some exports transpose; make it [num_boxes, 84] either way.
    preds = out.T if out.shape[0] == len(COCO_CLASSES) + 4 else out
    scores = preds[:, 4:]                 # [num_boxes, 80] class probabilities
    best = scores.max(axis=1)             # top class confidence per box
    cls_ids = scores.argmax(axis=1)       # which class that was
    present = {COCO_CLASSES[c] for c in cls_ids[best > conf].tolist()}
    return sorted(present)


# --- storage/query helpers: labels are stored as a delimited string so a plain
# SQL LIKE can test membership without matching substrings across classes
# (e.g. '|car|' won't match 'carrot'). ---
def labels_to_field(labels: list[str]) -> str:
    return "|" + "|".join(labels) + "|" if labels else "|"


def like_clause(cls: str) -> str:
    return f"labels LIKE '%|{cls}|%'"
