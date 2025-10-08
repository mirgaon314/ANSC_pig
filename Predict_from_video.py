#!/usr/bin/env python3
"""
Predict_from_video.py

Runs the full pipeline on a video:
  - Stage 1: detect pig + objects (single 'object' class), assign objects to left/right by vertical position (top=left, bottom=right).
  - Stage 2: crop pig, detect only "head".
  - Build the same feature vector used during training.
  - Load the trained multimodal fusion checkpoint and predict:
        not_interacting | interacting_left | interacting_right
  - Write an annotated MP4 and (optionally) a per-frame CSV of predictions.

Usage:
  python ../ANSC/ANSC_pig/Predict_from_video.py \
      --video ../ANSC/5min/Archive/F003_5min.mp4 \
      --pigobj ../ANSC/ANSC_pig/models/yolo_pig_object/weights/best.pt \
      --head ../ANSC/ANSC_pig/models/yolo_headtail/weights/best.pt \
      --mm-ckpt cls_runs_mm/mm_fusion_best.pt \
      --imgsz 224 \
      --device mps \
      --outvid ../ANSC/detected/F003_pred.mp4 \
      --outcsv ../ANSC/detected/F003_pred.csv
"""

import os
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'  # safe on Mac/MPS

import argparse
import sys
from pathlib import Path
import math
import csv
import numpy as np
import cv2
from ultralytics import YOLO
import pandas as pd

import torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image

# Import marker builder
from markers_from_prediction import build_marker_table_from_prediction

# --- feature names must match Train_classification.py ---
BASE_FEATURES = [
    "pig_cx","pig_cy","pig_w","pig_h",
    "left_object_cx","left_object_cy","left_object_w","left_object_h",
    "right_object_cx","right_object_cy","right_object_w","right_object_h",
    "head_cx","head_cy","head_w","head_h",
]
DERIVED_FEATURES = [
    "dist_pig_to_left","dist_pig_to_right",
    "dist_head_to_left","dist_head_to_right",
    "dx_head_left","dy_head_left","dx_head_right","dy_head_right",
    "dx_pig_left","dy_pig_left","dx_pig_right","dy_pig_right",
    "pig_aspect","head_aspect","left_object_aspect","right_object_aspect",
]
ALL_FEATURES = BASE_FEATURES + DERIVED_FEATURES

def euclid(ax, ay, bx, by):
    if any([v is None or np.isnan(v) for v in (ax, ay, bx, by)]):
        return np.nan
    return math.hypot(ax - bx, ay - by)

def aspect(w,h):
    if w in (None,0) or h in (None,0) or any(np.isnan(v) for v in (w,h)):
        return np.nan
    return float(w)/float(h)

def center_w_h(x1,y1,x2,y2):
    w = max(0, x2-x1); h = max(0, y2-y1)
    return (x1+x2)/2.0, (y1+y2)/2.0, w, h

def choose_side_from_geometry(feat_row: dict) -> str:
    """
    Decide left/right by whichever object center is closer to the head center.
    Fall back to pig center if head is missing. Uses existing euclid().
    """
    hx, hy = feat_row.get("head_cx"), feat_row.get("head_cy")
    px, py = feat_row.get("pig_cx"),  feat_row.get("pig_cy")
    lx, ly = feat_row.get("left_object_cx"),  feat_row.get("left_object_cy")
    rx, ry = feat_row.get("right_object_cx"), feat_row.get("right_object_cy")

    def _is_nan(v):
        return v is None or (isinstance(v, float) and np.isnan(v))

    use_head = not (_is_nan(hx) or _is_nan(hy))
    if use_head:
        dL = euclid(hx, hy, lx, ly)
        dR = euclid(hx, hy, rx, ry)
    else:
        dL = euclid(px, py, lx, ly)
        dR = euclid(px, py, rx, ry)

    if np.isnan(dL): dL = float("inf")
    if np.isnan(dR): dR = float("inf")
    return "interacting_left" if dL <= dR else "interacting_right"

def crop_to_canvas(frame, x1, y1, x2, y2, canvas=480, bg=(114,114,114)):
    """Crop [x1:x2, y1:y2] and letterbox onto a square canvas without distortion."""
    import numpy as np
    h, w = frame.shape[:2]
    x1 = max(0, min(w - 1, int(x1)))
    y1 = max(0, min(h - 1, int(y1)))
    x2 = max(0, min(w,     int(x2)))
    y2 = max(0, min(h,     int(y2)))
    if x2 <= x1 or y2 <= y1:
        return None
    crop = frame[y1:y2, x1:x2].copy()
    ch, cw = crop.shape[:2]
    scale = min(canvas / max(1, cw), canvas / max(1, ch))
    new_w = max(1, int(round(cw * scale)))
    new_h = max(1, int(round(ch * scale)))
    resized = cv2.resize(crop, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    canvas_img = np.full((canvas, canvas, 3), bg, dtype=np.uint8)
    off_x = (canvas - new_w) // 2
    off_y = (canvas - new_h) // 2
    canvas_img[off_y:off_y + new_h, off_x:off_x + new_w] = resized
    return canvas_img

# ---- Multimodal Fusion model (must match Train_multimodal.py) ----
class FusionNet(nn.Module):
    def __init__(self, n_tab: int, n_classes: int = 3, img_embed: int = 256, tab_embed: int = 128):
        super().__init__()
        from torchvision.models import resnet18, ResNet18_Weights
        backbone = resnet18(weights=ResNet18_Weights.DEFAULT)
        in_feats = backbone.fc.in_features
        backbone.fc = nn.Identity()
        self.backbone = backbone
        self.img_head = nn.Linear(in_feats, img_embed)
        self.tab_mlp = nn.Sequential(
            nn.Linear(n_tab, 256),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(256),
            nn.Dropout(0.10),
            nn.Linear(256, tab_embed),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(tab_embed),
        )
        self.classifier = nn.Sequential(
            nn.Linear(img_embed + tab_embed, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.25),
            nn.Linear(128, n_classes),
        )
    def forward(self, img, tab):
        z_img = self.backbone(img)
        z_img = self.img_head(z_img)
        z_tab = self.tab_mlp(tab)
        z = torch.cat([z_img, z_tab], dim=1)
        return self.classifier(z)

def assign_objects_vertical(objs, H):
    """
    objs: list of (x1,y1,x2,y2,conf)
    returns: dict with (left_*) and (right_*) fields based on center-y (top=left, bottom=right)
    """
    out = {
        "left_object_cx": np.nan, "left_object_cy": np.nan, "left_object_w": np.nan, "left_object_h": np.nan, "left_object_conf": np.nan,
        "right_object_cx": np.nan, "right_object_cy": np.nan, "right_object_w": np.nan, "right_object_h": np.nan, "right_object_conf": np.nan,
    }
    if not objs:
        return out
    # sort by center-y
    with_cy = []
    for (x1,y1,x2,y2,conf) in objs:
        cx, cy, w, h = center_w_h(x1,y1,x2,y2)
        with_cy.append((cy, (cx,cy,w,h,conf)))
    with_cy.sort(key=lambda t: t[0])  # top..bottom

    if len(with_cy) >= 2:
        (cx,cy,w,h,conf) = with_cy[0][1]
        out.update({"left_object_cx":cx,"left_object_cy":cy,"left_object_w":w,"left_object_h":h,"left_object_conf":conf})
        (cx,cy,w,h,conf) = with_cy[-1][1]
        out.update({"right_object_cx":cx,"right_object_cy":cy,"right_object_w":w,"right_object_h":h,"right_object_conf":conf})
    else:
        (cx,cy,w,h,conf) = with_cy[0][1]
        if cy < H/2:
            out.update({"left_object_cx":cx,"left_object_cy":cy,"left_object_w":w,"left_object_h":h,"left_object_conf":conf})
        else:
            out.update({"right_object_cx":cx,"right_object_cy":cy,"right_object_w":w,"right_object_h":h,"right_object_conf":conf})
    return out

def build_feature_row(pig_box, head_box, left_obj, right_obj):
    """
    pig_box/head_box: (cx,cy,w,h,conf) or None
    left/right_obj:   dict fields already cx,cy,w,h,conf (may be NaN)
    returns dict of ALL_FEATURES filled with float (NaN allowed)
    """
    row = {k: np.nan for k in ALL_FEATURES}
    # pig
    if pig_box:
        pcx,pcy,pw,ph,pconf = pig_box
        row.update({"pig_cx":pcx,"pig_cy":pcy,"pig_w":pw,"pig_h":ph})
    # head
    if head_box:
        hcx,hcy,hw,hh,hconf = head_box
        row.update({"head_cx":hcx,"head_cy":hcy,"head_w":hw,"head_h":hh})
    # objects
    for side in ("left","right"):
        ocx = left_obj["left_object_cx"] if side=="left" else right_obj["right_object_cx"]
        ocy = left_obj["left_object_cy"] if side=="left" else right_obj["right_object_cy"]
        ow  = left_obj["left_object_w"]  if side=="left" else right_obj["right_object_w"]
        oh  = left_obj["left_object_h"]  if side=="left" else right_obj["right_object_h"]
        # base features
        row[f"{side}_object_cx"] = ocx
        row[f"{side}_object_cy"] = ocy
        row[f"{side}_object_w"] = ow
        row[f"{side}_object_h"] = oh

    # derived distances
    loc = left_obj
    roc = right_obj
    row["dist_pig_to_left"]  = euclid(row["pig_cx"], row["pig_cy"],  loc["left_object_cx"],  loc["left_object_cy"])
    row["dist_pig_to_right"] = euclid(row["pig_cx"], row["pig_cy"],  roc["right_object_cx"], roc["right_object_cy"])
    row["dist_head_to_left"] = euclid(row["head_cx"], row["head_cy"], loc["left_object_cx"],  loc["left_object_cy"])
    row["dist_head_to_right"]= euclid(row["head_cx"], row["head_cy"], roc["right_object_cx"], roc["right_object_cy"])

    # deltas
    row["dx_head_left"]  = (row["head_cx"] - loc["left_object_cx"])  if not np.isnan(row["head_cx"])  and not np.isnan(loc["left_object_cx"]) else np.nan
    row["dy_head_left"]  = (row["head_cy"] - loc["left_object_cy"])  if not np.isnan(row["head_cy"])  and not np.isnan(loc["left_object_cy"]) else np.nan
    row["dx_head_right"] = (row["head_cx"] - roc["right_object_cx"]) if not np.isnan(row["head_cx"])  and not np.isnan(roc["right_object_cx"]) else np.nan
    row["dy_head_right"] = (row["head_cy"] - roc["right_object_cy"]) if not np.isnan(row["head_cy"])  and not np.isnan(roc["right_object_cy"]) else np.nan

    row["dx_pig_left"]   = (row["pig_cx"] - loc["left_object_cx"])   if not np.isnan(row["pig_cx"])   and not np.isnan(loc["left_object_cx"]) else np.nan
    row["dy_pig_left"]   = (row["pig_cy"] - loc["left_object_cy"])   if not np.isnan(row["pig_cy"])   and not np.isnan(loc["left_object_cy"]) else np.nan
    row["dx_pig_right"]  = (row["pig_cx"] - roc["right_object_cx"])  if not np.isnan(row["pig_cx"])   and not np.isnan(roc["right_object_cx"]) else np.nan
    row["dy_pig_right"]  = (row["pig_cy"] - roc["right_object_cy"])  if not np.isnan(row["pig_cy"])   and not np.isnan(roc["right_object_cy"]) else np.nan

    # aspects
    row["pig_aspect"]         = aspect(row["pig_w"], row["pig_h"])
    row["head_aspect"]        = aspect(row["head_w"], row["head_h"])
    row["left_object_aspect"] = aspect(loc["left_object_w"],  loc["left_object_h"])
    row["right_object_aspect"]= aspect(roc["right_object_w"], roc["right_object_h"])

    return row

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", default="../ANSC/5min/Archive/F003_5min.mp4")
    ap.add_argument("--pigobj", default="../ANSC/ANSC_pig/models/yolo_pig_object/weights/best.pt", help="YOLO weights for pig+object")
    ap.add_argument("--head", default="../ANSC/ANSC_pig/models/yolo_headtail/weights/best.pt", help="YOLO weights for head/tail model")
    ap.add_argument("--outvid", default="../ANSC/detected/F003_pred_new.mp4", help="output annotated mp4")
    ap.add_argument("--outcsv", default="../ANSC/detected/F003_pred_new.csv", help="optional per-frame predictions CSV")
    ap.add_argument("--outmarkers", default="../ANSC/detected/F003_pred_markers_60fps_new.csv", help="optional markers CSV (start/stop per interaction)")
    ap.add_argument("--trial", default="F003", help="trial name to include in markers")
    ap.add_argument("--mm-ckpt", default="../ANSC/ANSC_pig/models/cls_runs_mm/mm_fusion_best.pt", help="path to mm_fusion_best.pt")
    ap.add_argument("--imgsz", type=int, default=224, help="inference image size for fusion model")
    ap.add_argument("--device", default="mps", choices=["cpu","mps","cuda","0","1","2"], help="device for fusion model")
    ap.add_argument("--step", type=int, default=1, help="process every Nth frame (speed)")
    ap.add_argument("--interact-thresh", type=float, default=0.50,
                    help="probability threshold for 'interacting' (binary fusion model)")
    args = ap.parse_args()

    if len(sys.argv) == 1:
        print("[AUTO] No CLI args provided; using built-in defaults (see --help to override).")

    # Device (respect CUDA index)
    if args.device in ("0","1","2") and torch.cuda.is_available():
        torch.cuda.set_device(int(args.device))
        device = torch.device("cuda")
    elif args.device == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
    elif args.device == "mps" and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"[INFO] Using device for fusion model: {device}")

    # Load models
    model_pigobj = YOLO(args.pigobj)
    model_head   = YOLO(args.head)

    # Find head class id
    head_id = next(k for k,v in model_head.names.items() if v == "head")

    # Load multimodal fusion model
    ckpt = torch.load(args.mm_ckpt, map_location=device)
    n_tab = int(ckpt.get("n_tab", len(ALL_FEATURES)))
    feat_names_ckpt = ckpt.get("feature_names", ALL_FEATURES)
    id2label = ckpt.get("label_map", {0:"not_interacting", 1:"interacting"})
    # load optional feature normalization stats
    feat_mean = ckpt.get("feat_mean", None)
    feat_std = ckpt.get("feat_std", None)
    if isinstance(feat_mean, list):
        feat_mean = np.array(feat_mean, dtype=np.float32)
    if isinstance(feat_std, list):
        feat_std = np.array(feat_std, dtype=np.float32)
    use_norm = (isinstance(feat_mean, np.ndarray) and isinstance(feat_std, np.ndarray)
                and feat_mean.shape[0] == len(feat_names_ckpt) and feat_std.shape[0] == len(feat_names_ckpt))
    if use_norm:
        print("[INFO] Using saved feature normalization (z-score) from checkpoint.")
    else:
        print("[INFO] No compatible feature normalization found in checkpoint; using raw features.")

    model_fusion = FusionNet(n_tab=n_tab, n_classes=len(id2label))
    model_fusion.load_state_dict(ckpt["model_state"])
    model_fusion.to(device).eval()

    # Image transform (must match training)
    tf_img = T.Compose([
        T.Resize((args.imgsz, args.imgsz)),
        T.ToTensor(),
        T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
    ])

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {args.video}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    W   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    # Compute video duration (for end marker)
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    if frame_count and frame_count > 0 and fps and fps > 0:
        video_duration_sec = float(frame_count / fps)
    else:
        video_duration_sec = None
    # Ensure output directories exist
    Path(args.outvid).parent.mkdir(parents=True, exist_ok=True)
    if args.outcsv:
        Path(args.outcsv).parent.mkdir(parents=True, exist_ok=True)
    if args.outmarkers:
        Path(args.outmarkers).parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(args.outvid, fourcc, fps, (W,H))
    if not out.isOpened():
        raise RuntimeError(f"Cannot write to: {args.outvid}")

    DILATE = 10     # pixels around pig box to include context
    CANVAS = 480    # letterbox canvas to mimic training crops

    csv_writer = None
    if args.outcsv:
        csvf = open(args.outcsv, "w", newline="")
        csv_writer = csv.writer(csvf)
        csv_writer.writerow(["frame","time_sec","pred_label","pred_conf"])

    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if frame_idx % args.step != 0:
            frame_idx += 1
            continue

        # --- Stage 1: detect pig & object(s) on full frame ---
        det = model_pigobj.predict(source=frame, stream=False)[0]
        pig_best = None  # (cx,cy,w,h,conf)
        object_list = [] # [(x1,y1,x2,y2,conf), ...]

        for box, cls, conf in zip(det.boxes.xyxy.tolist(),
                                  det.boxes.cls.tolist(),
                                  det.boxes.conf.tolist()):
            name = model_pigobj.names[int(cls)]
            x1,y1,x2,y2 = map(int, box)
            cx,cy,w,h = center_w_h(x1,y1,x2,y2)
            if name == "pig":
                if (pig_best is None) or (conf > pig_best[4]):
                    pig_best = (cx,cy,w,h,float(conf))
                cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,255), 2)
                cv2.putText(frame, "pig", (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 1)
            elif name == "object":
                object_list.append((x1,y1,x2,y2,float(conf)))

        # assign objects top=left, bottom=right
        obj_assign = assign_objects_vertical(object_list, H)
        # draw assigned boxes (optional)
        if not np.isnan(obj_assign["left_object_cx"]):
            x1 = int(obj_assign["left_object_cx"] - obj_assign["left_object_w"]/2)
            x2 = int(obj_assign["left_object_cx"] + obj_assign["left_object_w"]/2)
            y1 = int(obj_assign["left_object_cy"] - obj_assign["left_object_h"]/2)
            y2 = int(obj_assign["left_object_cy"] + obj_assign["left_object_h"]/2)
            cv2.rectangle(frame, (x1,y1), (x2,y2), (255,0,0), 2)
            cv2.putText(frame, "left_obj", (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 1)
        if not np.isnan(obj_assign["right_object_cx"]):
            x1 = int(obj_assign["right_object_cx"] - obj_assign["right_object_w"]/2)
            x2 = int(obj_assign["right_object_cx"] + obj_assign["right_object_w"]/2)
            y1 = int(obj_assign["right_object_cy"] - obj_assign["right_object_h"]/2)
            y2 = int(obj_assign["right_object_cy"] + obj_assign["right_object_h"]/2)
            cv2.rectangle(frame, (x1,y1), (x2,y2), (0,0,255), 2)
            cv2.putText(frame, "right_obj", (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1)

        # --- Stage 2: head on pig crop ---
        head_best = None
        if pig_best is not None:
            # convert pig center/size back to xyxy to crop
            pcx,pcy,pw,ph,_ = pig_best
            x1 = max(0, int(pcx - pw/2)); y1 = max(0, int(pcy - ph/2))
            x2 = min(W, int(pcx + pw/2)); y2 = min(H, int(pcy + ph/2))
            crop = frame[y1:y2, x1:x2]
            if crop.size > 0:
                det2 = model_head.predict(source=crop, stream=False, classes=[head_id])[0]
                if len(det2.boxes):
                    confs = det2.boxes.conf.tolist()
                    best_i = int(max(range(len(confs)), key=lambda i: confs[i]))
                    bx = det2.boxes.xyxy.tolist()[best_i]
                    hx1,hy1,hx2,hy2 = bx
                    # map to full frame
                    HX1, HY1 = int(hx1 + x1), int(hy1 + y1)
                    HX2, HY2 = int(hx2 + x1), int(hy2 + y1)
                    hcx,hcy,hw,hh = center_w_h(HX1,HY1,HX2,HY2)
                    head_best = (hcx,hcy,hw,hh,float(confs[best_i]))
                    cv2.rectangle(frame, (HX1,HY1), (HX2,HY2), (0,255,0), 2)
                    cv2.putText(frame, "head", (HX1, HY1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)

        # --- Build features & predict ---
        left_dict = {k: np.nan for k in obj_assign} ; right_dict = {k: np.nan for k in obj_assign}
        # split dict into two small dicts for function interface clarity
        left_only  = {k:v for k,v in obj_assign.items() if k.startswith("left_")}
        right_only = {k:v for k,v in obj_assign.items() if k.startswith("right_")}
        feat_row = build_feature_row(pig_best, head_best, left_only, right_only)

        # --- Build tabular features in ckpt order ---
        X = pd.DataFrame([feat_row])
        for col in ALL_FEATURES:
            if col not in X.columns: X[col] = np.nan
        X = X[feat_names_ckpt]
        # Replace NaN with 0 for tabular input (matches training)
        X = X.fillna(0)

        # Convert to numpy and apply normalization if saved in checkpoint
        Xv = X.values.astype(np.float32)
        if 'use_norm' in locals() and use_norm:
            Xv = (Xv - feat_mean) / np.maximum(feat_std, 1e-6)

        # Crop pig box with dilation and letterbox to CANVAS
        if pig_best is not None:
            pcx,pcy,pw,ph,_ = pig_best
            x1 = pcx - pw/2 - DILATE
            y1 = pcy - ph/2 - DILATE
            x2 = pcx + pw/2 + DILATE
            y2 = pcy + ph/2 + DILATE
            crop_img = crop_to_canvas(frame, x1, y1, x2, y2, canvas=CANVAS)
        else:
            crop_img = np.full((CANVAS, CANVAS, 3), 114, dtype=np.uint8)

        if crop_img is None:
            # fallback to blank image if crop failed
            crop_img = np.full((CANVAS, CANVAS, 3), 114, dtype=np.uint8)

        # Convert crop_img to PIL Image and apply transform
        pil_img = Image.fromarray(cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB))
        img_tensor = tf_img(pil_img).unsqueeze(0).to(device)

        tab_tensor = torch.tensor(Xv, dtype=torch.float32, device=device)

        with torch.no_grad():
            logits = model_fusion(img_tensor, tab_tensor)
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]

        # find index for 'interacting' in ckpt map (fallback to class 1)
        inter_idx = None
        for k, v in id2label.items():
            if str(v) == "interacting":
                inter_idx = int(k)
                break
        if inter_idx is None and len(probs) >= 2:
            inter_idx = 1
        elif inter_idx is None:
            inter_idx = int(np.argmax(probs))

        p_interact = float(probs[inter_idx])

        if p_interact < args.interact_thresh:
            final_label = "not_interacting"
            conf = 1.0 - p_interact
        else:
            final_label = choose_side_from_geometry(feat_row)
            conf = p_interact

        # annotate frame with decision
        t = frame_idx / fps
        label_text = f"{final_label} (p_int={p_interact:.2f})"
        cv2.putText(frame, label_text, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (50,255,50), 2)

        out.write(frame)
        if csv_writer:
            csv_writer.writerow([frame_idx, f"{t:.2f}", final_label, f"{conf:.3f}"])

        frame_idx += 1

    cap.release()
    out.release()
    if csv_writer:
        csvf.close()
    # Build markers CSV from prediction CSV (if requested)
    if args.outcsv and args.outmarkers:
        fps_int = int(round(fps)) if fps and fps > 0 else 60
        # Infer trial name from arg or video filename
        trial_name = args.trial or Path(args.video).stem.split("_")[0]
        try:
            markers = build_marker_table_from_prediction(
                args.outcsv, args.outmarkers, trial_name, fps_int,
                video_duration_sec=video_duration_sec,
                gap_sec=0.25, min_dur_sec=0.05,
                force_start_at_zero=True, force_end_at_video_duration=True,
            )
            print(f"✅ Wrote markers CSV: {args.outmarkers}  (rows={len(markers)})")
        except Exception as e:
            print(f"[WARN] Failed to write markers CSV: {e}")
    print("✅ Wrote:", args.outvid)
    if args.outcsv:
        print("✅ Wrote:", args.outcsv)

if __name__ == "__main__":
    main()