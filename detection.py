import os
import cv2
from ultralytics import YOLO

# input_path  = "../ANSC/5min/Archive/B001_5min.mp4"
input_path = "../ANSC/testing.mp4"
output_path = "testing_.mp4"
cap = cv2.VideoCapture(input_path)
if not cap.isOpened():
    raise RuntimeError(f"Cannot open input video at {input_path}")

fps = cap.get(cv2.CAP_PROP_FPS)
w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('testing_.mp4', fourcc, fps, (w,h))

if not out.isOpened():
    raise RuntimeError(f"VideoWriter failed to open for: {output_path!r}")

model = YOLO("yolo11n.pt")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # run inference
    results = model.predict(source=frame, show_labels=False, stream=False)[0]
    annotated = results.plot()      # draws bboxes on `frame`
    out.write(annotated)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 6) Release and confirm
cap.release()
out.release()
cv2.destroyAllWindows()
print(f"Successfully saved annotated video to: {os.path.abspath(output_path)}")
