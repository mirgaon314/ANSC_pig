# extract_frames.py
import cv2, os

def extract_frames(video_path, out_dir="frames", step=60):
    os.makedirs(out_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    idx = 0
    saved = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # save every `step`th frame
        if idx % step == 0:
            fname = f"frame{idx:06d}.jpg"
            cv2.imwrite(os.path.join(out_dir, fname), frame)
            saved += 1
        idx += 1
    cap.release()
    print(f"Extracted {saved} frames to {out_dir}/")

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("video", help="Path to input video")
    p.add_argument("--out", default="frames", help="Output folder")
    p.add_argument("--step", type=int, default=5, help="Save every Nth frame")
    args = p.parse_args()
    extract_frames(args.video, args.out, args.step)

# python extract_frames.py ../5min/Archive/C003_5min.mp4 --out pseudo_frames --step 60

# export LABEL_STUDIO_LOCAL_FILES_SERVING_ENABLED=true
# export LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT="/Users/owen/Documents/College/Lab/ANSC/ANSC_pig"
