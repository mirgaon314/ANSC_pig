import pandas as pd
import numpy as np

try:
    import cv2
    from ultralytics import YOLO
except Exception:
    cv2 = None
    YOLO = None

try:
    import torch
except Exception:
    torch = None

# -----------------------------
# Device normalization helper
# -----------------------------

def _normalize_device(dev):
    """Return a YOLO/torch-friendly device or None.
    Accepts "", None, "cpu", "0", "1", "cuda", "cuda:0" etc.
    Falls back to CPU if CUDA is requested but not available.
    """
    if not dev:
        return None  # let YOLO decide (defaults to CPU)
    s = str(dev).strip().lower()
    # Map bare indices to cuda:n
    if s.isdigit():
        s = f"cuda:{s}"
    if s in {"gpu", "cuda"}:
        s = "cuda:0"
    if s == "cpu":
        return "cpu"
    # For any cuda:* request, verify availability
    if s.startswith("cuda"):
        if (torch is None) or (not torch.cuda.is_available()):
            print("[WARN] CUDA requested but not available. Falling back to CPU.")
            return "cpu"
        return s
    # Pass-through (e.g., "mps" on Apple Silicon, or already-normalized strings)
    return s
 
# -----------------------------
# Robust CSV loader
# -----------------------------
 
def load_markers_csv(path):
    """Load the markers CSV exported from annotation/video tools.
    Handles UTF-16/UTF-8 encodings and tab/comma separators, and skips a title row if present.
    Returns a pandas DataFrame with at least columns: 'Marker Name', 'In', 'Out'.
    """
    encodings = ["utf-16", "utf-16-le", "utf-16-be", "utf-8-sig", "latin1"]
    # Prefer tab first (common for UTF-16 exports), then let pandas sniff
    seps = ["\t", ",", None]
 
    last_err = None
    for enc in encodings:
        for sep in seps:
            try:
                df = pd.read_csv(path, encoding=enc, sep=sep, engine="python")
                # Normalize column names (strip whitespace / BOM)
                df.columns = [str(c).strip().replace("\ufeff", "") for c in df.columns]
                # If the first row is a single-cell title (e.g., "B010"), re-read skipping that row
                if ("Marker Name" not in df.columns) and df.shape[1] == 1:
                    df = pd.read_csv(path, encoding=enc, sep=sep, engine="python", skiprows=1)
                    df.columns = [str(c).strip().replace("\ufeff", "") for c in df.columns]
                # If still missing expected columns, try skipping one header row
                if "Marker Name" not in df.columns and "In" not in df.columns and "Out" not in df.columns:
                    # Attempt with skiprows=1 in case of an extra header/title row
                    df2 = pd.read_csv(path, encoding=enc, sep=sep, engine="python", skiprows=1)
                    df2.columns = [str(c).strip().replace("\ufeff", "") for c in df2.columns]
                    if "Marker Name" in df2.columns and "In" in df2.columns and "Out" in df2.columns:
                        df = df2
                # Final check
                if "Marker Name" in df.columns and "In" in df.columns and "Out" in df.columns:
                    return df
            except Exception as e:
                last_err = e
                continue
    # If we land here, raise the last error we saw with context
    raise RuntimeError(f"Failed to read CSV '{path}'. Last error: {last_err}")
 
# -----------------------------
# Timecode utilities
# -----------------------------
 
def time_to_frames(time_str, fps=60):
    """Convert timecode HH:MM:SS:FF into absolute frame number.
    Assumes the last field FF is a frame count at the given fps.
    """
    if pd.isna(time_str):
        return np.nan
    time_str = str(time_str).strip()
    h, m, s, f = map(int, time_str.split(":"))
    return ((h * 3600 + m * 60 + s) * fps) + f

# -----------------------------
# Detection utilities (optional)
# -----------------------------

def detect_coords_for_frames(video_path, frame_indices, model_pig_obj_path=None, model_head_path=None,
                              class_map=("pig","object","head"), imgsz=640, conf=0.25, iou=0.7, device=""):
    """Run detection on a set of frame indices and return a DataFrame with coordinates.
    Returns columns: Frame, pig_*, left_object_*, right_object_*, head_* (cx,cy,w,h,conf). 
    If detection libs are unavailable or models/video not provided, returns an empty DataFrame.
    """
    if (cv2 is None) or (YOLO is None):
        return pd.DataFrame()
    if not video_path or not model_pig_obj_path:
        return pd.DataFrame()

    pig_label, obj_label, head_label = class_map

    # Load models
    model1 = YOLO(model_pig_obj_path)
    _norm = _normalize_device(device)
    if _norm:
        model1.to(_norm)
    model2 = YOLO(model_head_path) if model_head_path else None
    if model2 and _norm:
        model2.to(_norm)

    # Build class-name -> id map for model1
    names1 = model1.model.names if hasattr(model1, "model") and hasattr(model1.model, "names") else model1.names
    name_to_id1 = {n: i for i, n in (names1.items() if isinstance(names1, dict) else enumerate(names1))}
    pig_id = name_to_id1.get(pig_label)
    obj_id = name_to_id1.get(obj_label)
    # Build class-name -> id map for model2 (head detection)
    if model2:
        names2 = model2.model.names if hasattr(model2, "model") and hasattr(model2.model, "names") else model2.names
        name_to_id2 = {n: i for i, n in (names2.items() if isinstance(names2, dict) else enumerate(names2))}
        head_id = name_to_id2.get(head_label)
    else:
        head_id = None

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    rows = []
    for fidx in sorted(set(int(x) for x in frame_indices if pd.notna(x))):
        # seek to frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, fidx)
        ok, frame = cap.read()
        if not ok:
            continue

        h, w = frame.shape[:2]

        res_list = model1.predict(source=frame, imgsz=imgsz, conf=conf, iou=iou, verbose=False,
                                  device=_normalize_device(device))
        if not res_list:
            rows.append({"Frame": fidx}); continue
        res = res_list[0]

        best_pig = None   # (x1,y1,x2,y2,conf)

        objects = []  # list of (cy, (x1,y1,x2,y2,conf))

        if res.boxes is not None and len(res.boxes) > 0:
            xyxy = res.boxes.xyxy.tolist()
            clss = res.boxes.cls.tolist()
            confs = res.boxes.conf.tolist()
            for box, cls_id, c in zip(xyxy, clss, confs):
                x1, y1, x2, y2 = map(float, box)
                if pig_id is not None and int(cls_id) == pig_id:
                    if (best_pig is None) or (c > best_pig[4]):
                        best_pig = (x1, y1, x2, y2, float(c))
                if obj_id is not None and int(cls_id) == obj_id:
                    cy = (y1 + y2) / 2.0
                    objects.append((cy, (x1, y1, x2, y2, float(c))))

        # Assign left_object and right_object based on vertical position
        left_object = None
        right_object = None
        if len(objects) >= 2:
            objects.sort(key=lambda x: x[0])  # sort by cy ascending
            left_object = objects[0][1]
            right_object = objects[-1][1]
        elif len(objects) == 1:
            cy, box = objects[0]
            if cy < (h / 2):
                left_object = box
            else:
                right_object = box

        best_head = None
        if best_pig is not None and model2 is not None and head_id is not None:
            x1, y1, x2, y2, _ = best_pig
            x1i, y1i, x2i, y2i = map(lambda v: max(0, int(round(v))), (x1, y1, x2, y2))
            crop = frame[y1i:y2i, x1i:x2i]
            if crop.size > 0:
                res2_list = model2.predict(source=crop, imgsz=imgsz, conf=conf, iou=iou, verbose=False,
                                           device=_normalize_device(device))
                if res2_list:
                    res2 = res2_list[0]
                    if res2.boxes is not None and len(res2.boxes) > 0:
                        xyxy2 = res2.boxes.xyxy.tolist()
                        clss2 = res2.boxes.cls.tolist()
                        confs2 = res2.boxes.conf.tolist()
                        # Filter boxes by head_id
                        head_boxes = [(i, confs2[i], xyxy2[i]) for i in range(len(clss2)) if int(clss2[i]) == head_id]
                        if head_boxes:
                            # Select best head by highest confidence
                            idx_best, _, hb = max(head_boxes, key=lambda x: x[1])
                            hx1 = float(hb[0] + x1i); hy1 = float(hb[1] + y1i)
                            hx2 = float(hb[2] + x1i); hy2 = float(hb[3] + y1i)
                            best_head = (hx1, hy1, hx2, hy2, float(confs2[idx_best]))

        row = {"Frame": fidx}
        if best_pig is not None:
            px1, py1, px2, py2, pc = best_pig
            row.update({
                "pig_cx": (px1+px2)/2.0, "pig_cy": (py1+py2)/2.0,
                "pig_w": (px2-px1),     "pig_h": (py2-py1),
                "pig_conf": pc,
            })
        if left_object is not None:
            ox1, oy1, ox2, oy2, oc = left_object
            row.update({
                "left_object_cx": (ox1+ox2)/2.0, "left_object_cy": (oy1+oy2)/2.0,
                "left_object_w": (ox2-ox1),     "left_object_h": (oy2-oy1),
                "left_object_conf": oc,
            })
        if right_object is not None:
            ox1, oy1, ox2, oy2, oc = right_object
            row.update({
                "right_object_cx": (ox1+ox2)/2.0, "right_object_cy": (oy1+oy2)/2.0,
                "right_object_w": (ox2-ox1),     "right_object_h": (oy2-oy1),
                "right_object_conf": oc,
            })
        if best_head is not None:
            hx1, hy1, hx2, hy2, hc = best_head
            row.update({
                "head_cx": (hx1+hx2)/2.0, "head_cy": (hy1+hy2)/2.0,
                "head_w": (hx2-hx1),     "head_h": (hy2-hy1),
                "head_conf": hc,
            })
        rows.append(row)

    cap.release()
    return pd.DataFrame(rows)
 
# -----------------------------
# Main conversion
# -----------------------------
 
def make_per_frame_csv(input_csv, output_csv, fps=60, overlap="concat",
                       video_path=None,
                       model_pig_obj_path=None,
                       model_head_path=None,
                       class_map=("pig","object","head"),
                       imgsz=640, conf=0.25, iou=0.7, device=""):
    """Create a per-frame CSV with three columns: Frame, HasMarker, MarkerName.

    Parameters
    ----------
    input_csv : str
        Path to the marker CSV
    output_csv : str
        Path to write the per-frame CSV
    fps : int
        Frames per second used for the HH:MM:SS:FF timecodes
    overlap : {"concat", "first", "last"}
        Strategy if multiple markers overlap on the same frame:
        - "concat": join names with '|'
        - "first": keep the first encountered
        - "last": keep the last encountered
    video_path : str, optional
        Path to the video file for detection
    model_pig_obj_path : str, optional
        Path to YOLO model weights for pig/object detection
    model_head_path : str, optional
        Path to YOLO model weights for head detection
    class_map : tuple, optional
        Class names for pig, object, and head
    imgsz : int, optional
        Image size for YOLO detection
    conf : float, optional
        Confidence threshold for YOLO detection
    iou : float, optional
        IOU threshold for YOLO detection
    device : str, optional
        Device to run YOLO on (e.g., "0" for GPU)

    Returns
    -------
    Writes a CSV with columns including Frame, HasMarker, MarkerName, pig_*, left_object_*, right_object_*, head_*.
    """
    df = load_markers_csv(input_csv)
 
    # Drop rows without In/Out
    df = df.dropna(subset=["In", "Out"]).copy()
 
    # Compute frame indices
    df["start_frame"] = df["In"].apply(lambda x: time_to_frames(x, fps))
    df["end_frame"] = df["Out"].apply(lambda x: time_to_frames(x, fps))
 
    # Limit to valid rows
    df = df[(df["start_frame"].notna()) & (df["end_frame"].notna())]
 
    if df.empty:
        # Nothing to mark; emit an empty template with a single row
        out = pd.DataFrame({"Frame": [], "HasMarker": [], "MarkerName": []})
        out.to_csv(output_csv, index=False)
        return
 
    # Determine overall frame range (inclusive)
    min_frame = int(df["start_frame"].min())
    max_frame = int(df["end_frame"].max())
 
    frames = pd.DataFrame({"Frame": np.arange(min_frame, max_frame + 1, dtype=int)})
    frames["HasMarker"] = False
    frames["MarkerName"] = np.nan
    # Force object dtype to avoid FutureWarning when assigning strings
    frames["MarkerName"] = frames["MarkerName"].astype("object")
 
    # Fill marker info
    for _, row in df.iterrows():
        start, end = int(row["start_frame"]), int(row["end_frame"])  # inclusive
        name = row.get("Marker Name", None)
        if pd.isna(name):
            name = None
        else:
            name = str(name)
        mask = (frames["Frame"] >= start) & (frames["Frame"] <= end)
 
        if overlap == "concat":
            # If already has a name, append with '|'
            existing = frames.loc[mask, "MarkerName"].astype("object").copy()
            if name is None:
                # nothing to write, just flip the flag
                frames.loc[mask, "HasMarker"] = True
            else:
                # fill NaNs with the new name
                to_write = existing.where(existing.notna(), other=name)
                # append where existing already had a value
                both_mask = existing.notna()
                if both_mask.any():
                    # build combined strings safely
                    combined = existing[both_mask].astype(str) + "|" + name
                    to_write.loc[both_mask] = combined
                frames.loc[mask, "MarkerName"] = to_write.astype("object")
                frames.loc[mask, "HasMarker"] = True
        else:
            # first/last behavior
            if overlap == "first":
                upd_mask = mask & (~frames["HasMarker"])  # only fill empty slots
            else:  # "last"
                upd_mask = mask
            if name is not None:
                frames.loc[upd_mask, "MarkerName"] = name
            frames.loc[upd_mask, "HasMarker"] = True

    # Optionally run detection and merge per-frame coordinates
    if video_path and model_pig_obj_path:
        coords_df = detect_coords_for_frames(
            video_path=video_path,
            frame_indices=frames["Frame"].tolist(),
            model_pig_obj_path=model_pig_obj_path,
            model_head_path=model_head_path,
            class_map=class_map,
            imgsz=imgsz, conf=conf, iou=iou, device=device,
        )
        if not coords_df.empty:
            frames = frames.merge(coords_df, on="Frame", how="left")

    frames.to_csv(output_csv, index=False)
 
 
# Example usage
# 1) Convert markers -> per-frame only
# make_per_frame_csv("../ANSC/mark/C003.csv", "../ANSC/mark/C003_per_frame.csv", fps=60)

# 2) Convert and also append coords by running detection on a video
# Set just the recording name (e.g., "B001") and paths will be constructed automatically
from pathlib import Path

NAMES = ["FB001","FB002","FB003","FB004","FB005"]          # <-- change this only
BASE = Path("../ANSC") # base folder


for NAME in NAMES:
    input_csv  = BASE / "mark" / f"{NAME}.csv"
    output_csv = BASE / "mark_frame" / f"{NAME}_per_frame.csv"
    video_path = BASE / "rough" / f"{NAME}.mp4"

    make_per_frame_csv(
        input_csv = str(input_csv),
        output_csv = str(output_csv),
        fps = 60,
        video_path = str(video_path),
        model_pig_obj_path = "/Users/owen/runs/detect/yolo11n_finetune_custom/weights/best.pt",
        model_head_path    = "/Users/owen/runs/detect/yolo_headtail/weights/best.pt",
        class_map=("pig","object","head"),  # adjust if your model names differ
        imgsz=640, conf=0.25, iou=0.7, device="mps"  # set device to GPU index if available
    )
