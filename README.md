# HRI30 Two-Stream Action Recognition

**Late-fusion RGB + skeleton action recognition for industrial human–robot interaction — 91.4% on 30 classes.**

A two-stream model that recognises 30 industrial human–robot interaction actions from the **HRI30** dataset (e.g. drilling / polishing while moving in eight directions, picking up & putting down tools, delivering objects, walking variants). An appearance stream and a pose stream are trained independently and combined by weighted late fusion.

## Results
- **Fusion accuracy: 91.4%** on the 30-class HRI30 set
- **+2.5%** over the best single stream (RGB + skeleton late fusion)
- Trained on an **8 GB GPU** via mixed precision + gradient accumulation

## Architecture

```
                 ┌─ RGB stream:   R(2+1)D-18 (Kinetics-400 pretrained) ─┐
 video (.avi) ───┤                                                      ├─ softmax ─┐
                 └─ Pose stream:  MediaPipe → CNN-BiLSTM-Attention ──────┘           │
                                                                                     ▼
                                          late fusion:  0.51·RGB + 0.49·Pose  →  action
```

**RGB / appearance stream** — `rgb_model.py`
- R(2+1)D-18 (factorised 3D CNN) pretrained on Kinetics-400, head re-trained for 30 classes.
- 16 frames uniformly sampled per clip, 128×128, Kinetics normalisation.
- 8 GB-friendly training: **automatic mixed precision (AMP)** + **gradient accumulation** (batch 4 × 4 steps = effective batch 16), SGD + ReduceLROnPlateau.

**Pose / skeleton stream** — `train_model_v3.py`
- **MediaPipe Pose** → 33 keypoints → root-centred (hip midpoint) and shoulder-distance scaled → 99-dim per frame, fixed 100-frame window.
- **Lightweight CNN-BiLSTM-Attention**: Conv1d feature extractor → 2-layer bidirectional LSTM (hidden 128) → multi-head self-attention (4 heads) → mean pool → classifier.
- Class-balanced loss; light augmentation (joint noise + random single-hand drop-out) to curb over-fitting.

**Skeleton extraction** — `batch_extract.py`
- Batch MediaPipe Pose over the video set; **forward-fill on detection loss** (occlusion) instead of zero-padding, preserving temporal continuity.

**Late fusion / inference** — `multimodel_fusion_new.py`
- Softmax each stream and combine with tuned weights (0.51 RGB / 0.49 pose); reports the fused prediction, confidence, latency and each stream's individual vote.

## Tech stack
Python · PyTorch · torchvision (R(2+1)D) · MediaPipe · OpenCV · scikit-learn

## Key scripts
```
batch_extract.py             # MediaPipe skeleton extraction → .npy
rgb_model.py                 # RGB stream: R(2+1)D-18 training (AMP + grad accumulation)
train_model_v3.py            # Pose stream: CNN-BiLSTM-Attention training
multimodel_fusion_new.py     # late fusion + real-time inference
predict_multimodel_final.py  # batch prediction
analyze_fusion.py            # fusion-weight analysis
```

## Notes
- Dataset (HRI30 videos / annotations) and trained weights (`*.pth`) are **not** included.
- Developed for the Sensing & Perception module, MSc Robotics, King's College London.
