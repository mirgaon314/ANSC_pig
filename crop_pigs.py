import os
import cv2
from ultralytics import YOLO

VIDEO_PATH    = "../ANSC/5min/Archive/C003_5min.mp4"
MODEL_WEIGHTS = "/Users/owen/runs/detect/yolo11n_finetune_custom/weights/best.pt"
OUT_DIR       = "../ANSC/detected/pig_crops"
STEP          = 120

os.makedirs(OUT_DIR, exist_ok=True)
model = YOLO(MODEL_WEIGHTS)

cap = cv2.VideoCapture(VIDEO_PATH)
frame_idx = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Only process every STEP-th frame
    if frame_idx % STEP == 0:
        # 1) Run pig+object inference on this frame
        res = model.predict(source=frame, stream=False)[0]

        # 2) Crop out each detected pig
        for box, cls in zip(res.boxes.xyxy.tolist(), res.boxes.cls.tolist()):
            if model.names[int(cls)] != "pig":
                continue
            x1, y1, x2, y2 = map(int, box)
            crop = frame[y1:y2, x1:x2]

            # 3) Save the crop with a name indicating the original frame
            out_name = f"frame{frame_idx:06d}.jpg"
            cv2.imwrite(os.path.join(OUT_DIR, out_name), crop)

    frame_idx += 1

cap.release()
print(f"Saved pig crops (every {STEP} frames) to {OUT_DIR}/")
