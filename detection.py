import os
import cv2
from ultralytics import YOLO

# source ~/envs/cv/bin/activate

input_path  = "../ANSC/5min/Archive/C003_5min.mp4"
# input_path = "../ANSC/testing.mp4"
output_path = "../ANSC/detected/C003_withhead.mp4"
cap = cv2.VideoCapture(input_path)
if not cap.isOpened():
    raise RuntimeError(f"Cannot open input video at {input_path}")

fps = cap.get(cv2.CAP_PROP_FPS)
w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
mid_x = w // 2

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (w,h))

if not out.isOpened():
    raise RuntimeError(f"VideoWriter failed to open for: {output_path!r}")

model1 = YOLO("/Users/owen/runs/detect/yolo11n_finetune_custom/weights/best.pt")
model2 = YOLO("/Users/owen/runs/detect/yolo_headtail/weights/best.pt")

frame_idx = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # === Stage 1: pig + object detection ===
    res1 = model1.predict(source=frame, stream=False)[0]
    pigs    = []
    objects = []

    for box, cls in zip(res1.boxes.xyxy.tolist(), res1.boxes.cls.tolist()):
        name = model1.names[int(cls)]
        x1, y1, x2, y2 = map(int, box)

        if name == "pig":
            pigs.append((x1, y1, x2, y2))
            # draw the pig body box (optional)
            cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,255), 2)
            cv2.putText(frame, "pig", (x1, y1-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 1)

        elif name == "object":
            cx = (x1 + x2) // 2
            side = "left" if cx < mid_x else "right"
            objects.append((x1,y1,x2,y2,side))
            color = (255,0,0) if side=="left" else (0,0,255)
            cv2.rectangle(frame, (x1,y1), (x2,y2), color, 2)
            cv2.putText(frame, f"{side}_obj", (x1,y1-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    # === Stage 2: only detect one head per pig ===
    head_id = [k for k,v in model2.names.items() if v == "head"][0]

    for (x1, y1, x2, y2) in pigs:
        pig_crop = frame[y1:y2, x1:x2]
        if pig_crop.size == 0:
            continue

        # only run head class
        res2 = model2.predict(
            source=pig_crop,
            stream=False,
            classes=[head_id]
        )[0]

        if res2.boxes:
            # pick the highest-confidence head
            # res2.boxes.conf is a tensor of confidences
            confs = res2.boxes.conf.tolist()
            idx_best = int(max(range(len(confs)), key=lambda i: confs[i]))
            hb = res2.boxes.xyxy.tolist()[idx_best]

            # map and draw
            hx1 = int(hb[0] + x1)
            hy1 = int(hb[1] + y1)
            hx2 = int(hb[2] + x1)
            hy2 = int(hb[3] + y1)
            cv2.rectangle(frame, (hx1,hy1), (hx2,hy2), (0,255,0), 2)
            cv2.putText(frame, "head", (hx1, hy1-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)


    # write annotated frame
    out.write(frame)
    frame_idx += 1

# 6) Release and confirm
cap.release()
out.release()
cv2.destroyAllWindows()
print(f"Successfully saved annotated video to: {os.path.abspath(output_path)}")
