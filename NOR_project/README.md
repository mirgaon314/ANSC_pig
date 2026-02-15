# NOR Project – Pig/Object Interaction Classifier

This repository trains a 5-frame convolutional model to determine whether a pig is interacting with an object in the centre frame. Each clip keeps the original 480×480 crops and pairs them with auxiliary temporal features.

## Prerequisites
- Python 3.10+
- Apple Silicon (preferred) to take advantage of `torch.mps`; CUDA and CPU fallbacks are supported.

Install dependencies:

```bash
python3 -m pip install -r requirements.txt
```

## Data Preparation
1. Generate balanced 5-frame sequences (already done once, re-run after changing preprocessing):
   ```bash
   python3 preprocess/create_balanced_sequences.py \
     --input-csv B004_per_frame_with_crops.csv \
     --output-csv balanced_sequences.csv \
     --image-root /Users/zimul3/Downloads/img
   ```
2. The script outputs:
   - `balanced_sequences.csv` – training manifest with frame paths and metadata.
   - `balanced_sequences.csv.summary.json` – class distribution report.

## Training
Run the PyTorch trainer (defaults to 15 epochs, focal loss, and cosine schedule):

```bash
python3 train.py \
  --balanced-csv balanced_sequences.csv \
  --metadata-csv B004_per_frame_with_crops.csv \
  --output-dir outputs \
  --batch-size 8 \
  --prefer-mps
```

Key flags:
- `--prefer-mps` enables Apple GPU acceleration when available (default).
- `--no-mps` falls back to CUDA/CPU.
- `--max-steps` and `--max-val-steps` limit batches per epoch for quick smoke tests.

Artifacts:
- `outputs/metrics.json` – per-epoch metrics.
- `outputs/best.pt` – best checkpoint by validation balanced accuracy.
- `outputs/last.pt` / `outputs/final.pt` – latest and final weights.

## Dry-Run Example
To verify the pipeline without a long run:

```bash
python3 train.py \
  --balanced-csv balanced_sequences.csv \
  --metadata-csv B004_per_frame_with_crops.csv \
  --output-dir debug_run \
  --epochs 1 \
  --max-steps 1 \
  --max-val-steps 1 \
  --batch-size 2
```

This executes a single optimisation step and writes metrics/checkpoints for inspection.

## Inference on a Folder of Frames

Generate interaction probabilities for every frame in a directory:

```bash
python3 predict.py \
  --images-dir /Users/zimul3/Downloads/img \
  --checkpoint outputs/full_run/best.pt \
  --output-csv predictions/full_run_predictions.csv \
  --prefer-mps
```

- Frames are processed in temporal order inferred from filename sorting.
- Each frame is evaluated with a 5-frame context window (edges are padded by repeating boundary frames).
- Adjust `--threshold` to change the 0/1 decision boundary (default 0.5).
- Use `--max-frames` for quick smoke tests before running on the entire folder.
