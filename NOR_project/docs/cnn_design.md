# 5-Frame Interaction Classifier – CNN Design

## Problem Framing
We classify whether the pig is interacting with the object in the **centre frame** of a 5-frame clip. Frames are sampled consecutively so that the clip covers two frames before and after the target frame. The model must be sensitive to sudden onsets (centre frame positive following negatives) and sustained interactions.

### Contract
- **Input**: Tensor $X \in \mathbb{R}^{5\times H\times W\times 3}$ (RGB frames ordered chronologically; frame index 2 is the target).
- **Output**: Scalar probability $p(y=1\mid X)$ with $y\in\{0,1\}$ indicating interaction.
- **Latency budget**: < 15 ms per clip on an RTX 3060 (batch size 32) after optimization.
- **Robustness**: Support missing/low-quality neighbour frames via masking and temporal dropout during training.

### Edge Cases to Cover
1. Clips where the centre frame is positive but the previous frames are negative (transition-on).
2. Clips with sustained contact where multiple consecutive frames are positive (steady interaction).
3. False positives caused by lighting changes or occlusions that resemble markers.
4. Motion blur or partial pig visibility.
5. Missing neighbour frames (e.g., at video boundaries).

## Input Preparation
- Keep crops at their native $480\times480$ resolution (no resizing) to preserve fine-grained marker detail.
- Normalise per-channel using ImageNet statistics to reuse pretrained weights.
- Inject two auxiliary channels at train time:
  1. **Temporal position encoding** (five scalar values broadcast spatially).
  2. **Interaction prior** (binary mask indicating known markers from metadata if available).
- Apply on-the-fly augmentations: random horizontal flip (if anatomy is symmetric with marker labels swapped), colour jitter, slight affine jitter ($\pm 5^\circ$), temporal dropout (replace one neighbour frame with the centre frame).

## CNN Backbone
Use a lightweight 2+1D CNN (spatial Conv2d followed by depthwise temporal Conv1d) with temporal squeeze-attention—this layout runs on MPS while approximating true 3D receptive fields:

1. **Stem**
   - Shared Conv2d (7×7, stride 2) applied per frame → 32 channels, BatchNorm, GELU.
   - Frame-wise MaxPool2d (stride 2) to downsample spatially only.
2. **Residual Temporal Blocks** (×3)
   - Each block: `[(Conv2d + BN + GELU) → (depthwise Conv1d + BN + GELU)] × 2` with identity skip.
   - Channel widths: 64, 128, 256 respectively. Spatial stride is 2 in stages 2 and 3; temporal stride stays at 1 to keep all 5 frames.
3. **Temporal Focus Module**
   - Compute attention weights $\alpha_t = \mathrm{softmax}(W^T h_t)$ from per-frame pooled features.
   - Force $\alpha_2$ (centre frame) to have a minimum weight via margin loss encouraging focus on the middle frame.
4. **Head**
   - Concatenate attention-weighted feature vector and centre-frame feature.
   - Dense(256) + GELU + Dropout(0.3).
   - Dense(64) + GELU + Dropout(0.2).
   - Dense(1) with sigmoid.

### Justification
- 2+1D separable convolutions capture short-term motion cues while remaining compatible with Apple `mps` acceleration (native Conv3D is unsupported).
- Attention ensures neighbouring frames contribute while centre frame remains dominant.
- Bottleneck-style residual blocks balance accuracy and speed. The network holds <5M parameters, trainable on a single GPU.
- Dropout and temporal dropout reduce overfitting to specific motion sequences.

## Training Strategy
- **Loss**: Binary cross-entropy with focal term $\gamma = 1.5$ to down-weight easy negatives.
- **Optimizer**: AdamW, $\eta = 2\times10^{-4}$, weight decay $1\times10^{-4}$.
- **Schedule**: Cosine decay with warmup over first 5 epochs.
- **Batch size**: 32 clips (adjust based on GPU memory).
- **Regularisation**: MixUp across batches (probability 0.2) applied on logits.

## Evaluation & Monitoring
- Stratified split ensuring clips from the same continuous video segment remain in the same fold to avoid leakage.
- Metrics: AUC, balanced accuracy, precision/recall at 0.5 threshold, and onset detection latency (frames between first positive GT and model positive).
- Use Grad-CAM on the centre frame to audit spatial focus.

## Deployment Notes
- Export to ONNX (dynamic batch) after folding BatchNorm.
- For edge inference, optionally replace Conv3D blocks with TensorRT-optimised kernels or convert to (2+1)D convolutions to reduce latency.
- Maintain a fallback rule-based detector (markers in metadata) for frames where neighbouring images are unavailable.
