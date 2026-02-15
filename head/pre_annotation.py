#!/usr/bin/env python3
"""
auto_annotate_to_labelstudio.py

Pre-annotate a folder of (already-cropped) pig images using a YOLO model, and
export:
  1) YOLO-format labels (labels/*.txt)
  2) Optional Label Studio JSON with predictions (pre-annotations)

Example:
  python auto_annotate_to_labelstudio.py \
    --input ../ANSC/mark_frame_crops/B004/img \
    --out   /ABS/PATH/to/auto_heads \
    --weights ../ANSC/ANSC_pig/models/yolo_headtail/weights/best.pt \
    --classes head \
    --make-ls-json \
    --image-root-url "/data/local-files/?d=auto_heads/images" \
    --device mps
"""

import os
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'  # safe on Mac/MPS

import argparse
from pathlib import Path
import json
import shutil

import cv2
import numpy as np
from ultralytics import YOLO


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def save_yolo_txt(label_fp: Path, cls_ids, xywhn):
    lines = []
    for c, (x, y, w, h) in zip(cls_ids, xywhn):
        x = float(np.clip(x, 0, 1))
        y = float(np.clip(y, 0, 1))
        w = float(np.clip(w, 0, 1))
        h = float(np.clip(h, 0, 1))
        lines.append(f"{int(c)} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n")
    with open(label_fp, "w") as f:
        f.writelines(lines)


def to_ls_rect_result(xyxy, img_w, img_h, label):
    """
    Convert pixel xyxy -> Label Studio RectangleLabels result (percent units).
    LS needs x,y,width,height in % of image width/height, plus original dims.
    """
    x1, y1, x2, y2 = xyxy
    x = max(0, x1)
    y = max(0, y1)
    w = max(0, x2 - x1)
    h = max(0, y2 - y1)
    return {
        "from_name": "label",        # must match your LS labeling config control name
        "to_name": "image",          # must match the image tag name in LS config
        "type": "rectanglelabels",
        "value": {
            "x": 100.0 * x / img_w,
            "y": 100.0 * y / img_h,
            "width": 100.0 * w / img_w,
            "height": 100.0 * h / img_h,
            "rotation": 0,
            "rectanglelabels": [label],
            "original_width": int(img_w),
            "original_height": int(img_h),
        },
        "score": None,               # you can set model confidence here if you want
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Folder with cropped images (scans recursively)")
    ap.add_argument("--out",   required=True, help="Output folder (will contain images/ and labels/)")
    ap.add_argument("--weights", required=True, help="YOLO weights (e.g., ../ANSC/ANSC_pig/models/yolo_headtail/weights/best.pt)")
    ap.add_argument("--classes", default="head",
                    help="Comma-separated class list (order must match training). Example: 'head' or 'head,tail'")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--iou", type=float, default=0.7)
    ap.add_argument("--device", default="mps", help="mps|cpu|cuda|0|1")
    ap.add_argument("--write-empty", action="store_true", help="Write empty .txt files when no boxes are found")
    # Label Studio JSON options
    ap.add_argument("--make-ls-json", action="store_true", help="Also write Label Studio tasks JSON with predictions")
    ap.add_argument("--ls-json-out", default="ls_tasks.json", help="Output JSON filename (inside --out)")
    ap.add_argument("--image-root-url", default=None,
                    help="URL prefix used in LS tasks (e.g., /data/local-files/?d=auto_heads/images). "
                         "If omitted, uses plain filenames; set LS Local Files envs accordingly.")
    args = ap.parse_args()

    input_dir = Path(args.input)
    out_dir   = Path(args.out)
    images_out = out_dir / "images"
    labels_out = out_dir / "labels"
    ensure_dir(images_out)
    ensure_dir(labels_out)

    # Build class map & write classes.txt
    class_list = [c.strip() for c in args.classes.split(",") if c.strip() != ""]
    classes_txt = out_dir / "classes.txt"
    with open(classes_txt, "w") as f:
        for c in class_list:
            f.write(c + "\n")

    # Collect images
    patterns = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tif", "*.tiff")
    img_paths = []
    for pat in patterns:
        img_paths.extend(input_dir.rglob(pat))
    img_paths = sorted(img_paths)

    if not img_paths:
        print(f"[auto] No images found under: {input_dir}")
        return

    # Load model
    model = YOLO(args.weights)

    # Prepare LS task list
    ls_tasks = []

    total, with_boxes = 0, 0
    for ip in img_paths:
        total += 1
        # Read image for dimensions
        img = cv2.imread(str(ip))
        if img is None:
            print(f"[warn] Could not read image: {ip}")
            continue
        H, W = img.shape[:2]

        # Predict
        res = model.predict(
            source=str(ip),
            conf=args.conf,
            iou=args.iou,
            imgsz=args.imgsz,
            device=args.device,
            verbose=False
        )[0]

        # Copy image to dataset
        dst_img = images_out / ip.name
        if str(ip.resolve()) != str(dst_img.resolve()):
            shutil.copy2(ip, dst_img)

        # Prepare YOLO label path
        label_fp = labels_out / (ip.stem + ".txt")

        if res.boxes is not None and len(res.boxes) > 0:
            b = res.boxes
            xywhn = b.xywhn.cpu().tolist()
            cls_ids = [int(c) for c in b.cls.cpu().tolist()]
            save_yolo_txt(label_fp, cls_ids, xywhn)
            with_boxes += 1
        else:
            if args.write-empty:
                open(label_fp, "w").close()

        # Build LS prediction entry if requested
        if args.make_ls_json:
            # Image URL/path for LS
            if args.image_root_url:
                image_url = f"{args.image_root_url}/{ip.name}"
            else:
                # Fallback: just the filename; configure LS local file serving to point to images_out
                image_url = str((images_out / ip.name).name)

            pred_results = []
            if res.boxes is not None and len(res.boxes) > 0:
                # For each bbox, add rectangle result (percent units)
                for cls_id, conf, xyxy in zip(
                        res.boxes.cls.cpu().tolist(),
                        res.boxes.conf.cpu().tolist(),
                        res.boxes.xyxy.cpu().tolist()):
                    label = class_list[int(cls_id)] if int(cls_id) < len(class_list) else f"class_{int(cls_id)}"
                    rr = to_ls_rect_result(xyxy, W, H, label)
                    rr["score"] = float(conf)
                    pred_results.append(rr)

            task = {
                "data": {"image": image_url},
                "predictions": [
                    {
                        "model_version": "yolo-prelabel",
                        "result": pred_results,
                        "score": float(np.mean(res.boxes.conf.cpu().numpy())) if (res.boxes is not None and len(res.boxes) > 0) else None,
                    }
                ]
            }
            ls_tasks.append(task)

    print(f"[auto] Done. {with_boxes}/{total} images had boxes.")
    print(f"Images  → {images_out}")
    print(f"Labels  → {labels_out}")
    print(f"Classes → {classes_txt}")

    if args.make_ls_json:
        ls_json_path = out_dir / args.ls_json_out
        with open(ls_json_path, "w") as f:
            json.dump(ls_tasks, f, indent=2)
        print(f"LS JSON → {ls_json_path}")
        print("\nImport tips for Label Studio:")
        print("  1) Set local-files serving:")
        print("     export LABEL_STUDIO_LOCAL_FILES_SERVING_ENABLED=true")
        print(f"     export LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT={out_dir.parent}")
        print("     label-studio")
        print("  2) Create a project with RectangleLabels control named 'label' and Image tag named 'image'.")
        print(f"  3) Import {ls_json_path.name}. Predictions will appear as pre-annotations.")
        if args.image_root_url:
            print(f"     Your image URLs are based on: {args.image_root_url}")

if __name__ == "__main__":
    main()