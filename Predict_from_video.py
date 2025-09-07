#!/usr/bin/env python3
"""
Predict_from_video.py

Runs the full pipeline on a video:
  - Stage 1: detect pig + objects (single 'object' class), assign objects to left/right by vertical position (top=left, bottom=right).
  - Stage 2: crop pig, detect only "head".
  - Build the same feature vector used during training.
  - Load the trained RandomForest pipeline (joblib pkl) and predict:
        not_interacting | interacting_left | interacting_right
  - Write an annotated MP4 and (optionally) a per-frame CSV of predictions.

Usage:
  python ../ANSC/ANSC_pig/Predict_from_video.py \
      --video ../ANSC/5min/Archive/F003_5min.mp4 \
      --pigobj ../ANSC/ANSC_pig/models/yolo_pig_object/weights/best.pt \
      --head ../ANSC/ANSC_pig/models/yolo_headtail/weights/best.pt \
      --clf ../ANSC/ANSC_pig/models/interaction_cls/interaction_cls_model.pkl \
      --outvid ../ANSC/detected/F003_pred.mp4 \
      --outcsv ../ANSC/detected/F003_pred.csv
"""

import os
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'  # safe on Mac/MPS

import argparse
import math
import csv
import numpy as np
import cv2
from ultralytics import YOLO
import joblib
import pandas as pd

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
    ap.add_argument("--video", required=True)
    ap.add_argument("--pigobj", required=True, help="YOLO weights for pig+object")
    ap.add_argument("--head", required=True, help="YOLO weights for head/tail model")
    ap.add_argument("--clf", required=True, help="joblib pkl from Train_classification.py")
    ap.add_argument("--outvid", required=True, help="output annotated mp4")
    ap.add_argument("--outcsv", default=None, help="optional per-frame predictions CSV")
    ap.add_argument("--step", type=int, default=1, help="process every Nth frame (speed)")
    args = ap.parse_args()

    # Load models
    model_pigobj = YOLO(args.pigobj)
    model_head   = YOLO(args.head)

    # Find head class id
    head_id = next(k for k,v in model_head.names.items() if v == "head")

    # Load classifier pipeline + feature list
    payload = joblib.load(args.clf)
    pipeline = payload["pipeline"]
    feat_names = payload["features"]

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {args.video}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    W   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(args.outvid, fourcc, fps, (W,H))
    if not out.isOpened():
        raise RuntimeError(f"Cannot write to: {args.outvid}")

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

        # align with training feature order
        X = pd.DataFrame([feat_row])
        for col in ALL_FEATURES:
            if col not in X.columns: X[col] = np.nan
        X = X[ALL_FEATURES]

        pred = pipeline.predict(X)[0]
        conf = None
        if hasattr(pipeline.named_steps["clf"], "predict_proba"):
            proba = pipeline.predict_proba(X)[0]
            conf = float(np.max(proba))

        # annotate frame with decision
        t = frame_idx / fps
        label_text = f"{pred} ({conf:.2f})" if conf is not None else f"{pred}"
        cv2.putText(frame, label_text, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (50,255,50), 2)

        out.write(frame)
        if csv_writer:
            csv_writer.writerow([frame_idx, f"{t:.2f}", pred, f"{conf:.3f}" if conf is not None else ""])

        frame_idx += 1

    cap.release()
    out.release()
    if csv_writer:
        csvf.close()
    print("✅ Wrote:", args.outvid)
    if args.outcsv:
        print("✅ Wrote:", args.outcsv)

if __name__ == "__main__":
    main()