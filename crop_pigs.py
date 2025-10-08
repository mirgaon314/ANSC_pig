import os
import cv2
import math
import pandas as pd
from collections import defaultdict
from pathlib import Path
import numpy as np

# ------------- CONFIG -------------
# List the recordings you want to process. Example: ["FB001","FB002", ...]
NAMES = ["B002","B003","B004","B005","C003","C004","C005","F002","F003","F004","F005","FB001","FB002","FB003","FB004","FB005"]  # <-- change this only

BASE = Path("../ANSC")  # base folder containing mark_frame/ and rough/
STEP = 1                # process every Nth frame (set >1 if you only want sparse crops)
DILATE = 10             # expand bbox by N pixels on each side
CANVAS = 480            # final square canvas size (pixels)
CANVAS_COLOR = (114, 114, 114)  # gray background
# ----------------------------------

def crop_to_canvas(frame, x1, y1, x2, y2, canvas=480, bg=(114,114,114)):
    """
    Crop frame[y1:y2, x1:x2], then letterbox onto a square canvas (canvas x canvas)
    without distortion, preserving aspect ratio.
    """
    h, w = frame.shape[:2]
    x1 = max(0, min(w - 1, int(x1)))
    y1 = max(0, min(h - 1, int(y1)))
    x2 = max(0, min(w,     int(x2)))
    y2 = max(0, min(h,     int(y2)))
    if x2 <= x1 or y2 <= y1:
        return None

    crop = frame[y1:y2, x1:x2].copy()
    ch, cw = crop.shape[:2]

    # scale to fit inside canvas
    scale = min(canvas / max(1, cw), canvas / max(1, ch))
    new_w = max(1, int(round(cw * scale)))
    new_h = max(1, int(round(ch * scale)))
    resized = cv2.resize(crop, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    # paste centered on gray canvas
    canvas_img = np.full((canvas, canvas, 3), bg, dtype=np.uint8)
    off_x = (canvas - new_w) // 2
    off_y = (canvas - new_h) // 2
    canvas_img[off_y:off_y + new_h, off_x:off_x + new_w] = resized
    return canvas_img

def process_one(name: str):
    """
    Process one recording:
      - reads per-frame CSV for pig coords
      - crops pig regions from the video
      - dilates bbox by DILATE px
      - pastes onto CANVAS x CANVAS gray image
      - saves to per-name folder
      - writes an updated CSV with crop paths
    """
    video_path = BASE / "rough" / f"{name}.mp4"
    input_csv  = BASE / "mark_frame" / f"{name}_per_frame.csv"
    # Per-name output folder structure:
    out_dir    = BASE / "mark_frame_crops" / name / "img"
    out_dir.mkdir(parents=True, exist_ok=True)
    output_csv = BASE / "mark_frame_crops" / name / f"{name}_per_frame_with_crops.csv"
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    if not video_path.exists():
        raise FileNotFoundError(f"[{name}] Video not found: {video_path}")
    if not input_csv.exists():
        raise FileNotFoundError(f"[{name}] Per-frame CSV not found: {input_csv}")

    # Load CSV and validate columns
    df = pd.read_csv(input_csv)
    col_map = {c.lower(): c for c in df.columns}
    required = ["frame", "pig_cx", "pig_cy", "pig_w", "pig_h"]
    for rc in required:
        if rc not in col_map:
            raise KeyError(f"[{name}] Required column missing in CSV: {rc} (columns present: {list(df.columns)})")
    # normalize column names
    Frame   = col_map["frame"]
    pig_cx  = col_map["pig_cx"]
    pig_cy  = col_map["pig_cy"]
    pig_w   = col_map["pig_w"]
    pig_h   = col_map["pig_h"]

    # Prepare output columns
    df["crop_dir"] = str(out_dir)
    df["crop_image"] = ""

    # Build set of frames to process (subsample via STEP)
    frames_to_process = set(df[Frame][::STEP].astype(int).tolist())

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"[{name}] Cannot open input video at {video_path}")

    # Iterate video sequentially; crop when frame index is in our set
    frame_idx = 0
    saved = 0
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

    # Create quick lookup for coords by frame
    df_idx = df.set_index(Frame)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx in frames_to_process:
            if frame_idx in df_idx.index:
                row = df_idx.loc[frame_idx]
                # read coordinates
                try:
                    cx = float(row[pig_cx]); cy = float(row[pig_cy])
                    bw = float(row[pig_w]);  bh = float(row[pig_h])
                except Exception:
                    # if multiple rows per frame, take first
                    if isinstance(row, pd.DataFrame):
                        r0 = row.iloc[0]
                        cx = float(r0[pig_cx]); cy = float(r0[pig_cy])
                        bw = float(r0[pig_w]);  bh = float(r0[pig_h])
                    else:
                        raise

                # Convert cx,cy,w,h (assumed pixels) to xyxy and dilate
                x1 = cx - bw / 2.0 - DILATE
                y1 = cy - bh / 2.0 - DILATE
                x2 = cx + bw / 2.0 + DILATE
                y2 = cy + bh / 2.0 + DILATE

                canvas_img = crop_to_canvas(frame, x1, y1, x2, y2, canvas=CANVAS, bg=CANVAS_COLOR)
                if canvas_img is not None:
                    out_name = f"frame{frame_idx:06d}.jpg"
                    out_path = out_dir / out_name
                    cv2.imwrite(str(out_path), canvas_img)
                    df.loc[df[Frame] == frame_idx, "crop_image"] = out_name
                    saved += 1

        frame_idx += 1

    cap.release()
    df.to_csv(output_csv, index=False)
    print(f"[OK] {name}: saved {saved} crops to {out_dir} and wrote {output_csv}")

def main():
    for name in NAMES:
        try:
            process_one(name)
        except Exception as e:
            print(f"[ERR] {name}: {e}")

if __name__ == "__main__":
    main()
