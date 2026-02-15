import os
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
import cv2
from ultralytics import YOLO
# source ls-venv/bin/activate

model = YOLO("yolo11n.pt")

# Train the model with MPS
results = model.train(
    data="../ANSC/ANSC_pig/head/head_v2_data/data.yaml",  # ← your new data file
    epochs=30,                     # number of epochs you want
    imgsz=640,                      # image size
    batch=16,                       # batch size
    device="mps",                   # on Mac GPU
    name="yolo_head_v2", # run name for outputs
    augment=True                    # keep YOLO’s mosaic/mixup on
)
