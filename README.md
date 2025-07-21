# Grad‑CAM Comparison Pipeline for PHOENIX‑2014

---

## 0  Purpose

Build **one Python‑3.8 / PyTorch‑1.13 codebase** that, with *zero manual tweaks*, can:

1. Load any PHOENIX‑2014 clip selected by the user (train / dev / test).
2. Run it through two pre‑trained CSLR checkpoints:

   * **SlowFast‑Sign** – `checkpoints/slowfast_phoenix2014_dev_18.01_test_18.28.pt`
   * **TwoStream‑SLR** – `checkpoints/twostreamslr.ckpt`
3. Extract Grad‑CAM heat‑maps from five predefined layers in each model.
4. Overlay those heat‑maps on the original frames and save per‑layer PNGs **and** an MP4 side‑by‑side comparison video.
5. Print the predicted gloss sequence from each model next to the ground‑truth (if available).

Everything must succeed on **one A40 (48 GB)** *or* **one A100 (80 GB)**, using mixed‑precision forwards and FP32 grads.

---

## 1  Runtime Environment

```bash
conda create -n gradcam38 python=3.8 -y
conda activate gradcam38

# core
pip install torch==1.13.0+cu117 torchvision==0.14.0+cu117 -f https://download.pytorch.org/whl/torch_stable.html

# utilities
pip install pytorch-grad-cam==1.5.2 opencv-python==4.10.0.82 av==10.0.0 numpy==1.24.4 pyyaml tqdm
```

No additional system packages are required; all image I/O is done via **PyAV** & **OpenCV**.

---

## 2  Directory Layout (fixed and Git setup)

```text
~/nirmal/gradcam/
├── checkpoints/
│   ├── slowfast_phoenix2014_dev_18.01_test_18.28.pt
│   └── twostreamslr.ckpt
├── code/                # ← **Git repository root** (already `git init`‑ed)
│   ├── datasets/ …      # Codex: place *all* .py files **directly here**
│   └── (no extra nesting)
├── data/                # ← **outside the repo, never pushed**
│   └── phoenix-2014-multisigner/
│       ├── annotations/manual/{train,dev,test}.corpus.csv
│       ├── annotations/manual/gloss2ids.pkl
│       ├── features/
│       │   ├── fullFrame-256x256px/{train,dev,test}/<seq>/1/*.png   # RGB
│       │   └── fullFrame-210x260px/{train,dev,test}/<seq>/1/*.png   # low‑res RGB for key‑points
│       └── keypoints/
│           ├── keypoints.pkl                     # [T,133,3] coords
│           └── heatmaps/                         # auto‑cached tensors
└── outputs/   # auto‑generated at runtime
```

**Important for Codex**

* Treat `~/nirmal/gradcam/code` as the **project root** – all modules, entry‑points, and package imports live here.
* **Do not create another `code/` folder** inside the repo; the existing directory is already version‑controlled.
* Paths to checkpoints/data remain **relative to this layout** (e.g. `../data/phoenix-2014-multisigner/...`).

---

## 3  Model Wrappers  Model Wrappers

### 3.1  SlowFast‑Sign

* **Backbone**: two ResNet‑3D streams

  * Slow path: `conv2d.slow_path.layer{1‑4}` → 2048‑c
  * Fast path: `conv2d.fast_path.layer{1‑4}` → 256‑c, α = 4 temporal stride
* **Fuse**: `conv1d.fused_features` (Add + BN + ReLU)
* **Temporal**: `temporal_model.{0‑2}` BiLSTM stack (1024 → 2048)
* **Classifier**: `classifier.{0‑2}` NormLinear, per‑frame 1296 gloss logits

#### CAM target layers

```
1. conv2d.slow_path.layer1
2. conv2d.fast_path.layer1
3. conv2d.fast_path.layer3
4. conv1d.fused_features
5. temporal_model.1        # 2nd Bi‑LSTM output (before CTC)
```

### 3.2  TwoStream‑SLR

* **Backbones**: two S3D streams

  * RGB stream: `rgb_stream.layer{1‑4}` (64,192,480,832‑c)
  * Pose stream: `pose_stream.layer{1‑4}` (same channels), fed with **2‑channel heat‑maps**
* **Fusion**: `fusion_module.conv_fuse` (bi‑directional lateral, k = (7,3,3))
* **Heads**:

  * VisualHead @ 832‑c for both streams (`visual_head`, `visual_head_keypoint`)
  * CTC loss: `recognition_loss_func`

#### CAM target layers

```
1. rgb_stream.layer1
2. pose_stream.layer1
3. fusion_module.conv_fuse
4. rgb_stream.layer4
5. visual_head            # logits before CTC
```

---

## 4  Unified Dataset ― `PhoenixVideoTextDataset`

```python
class PhoenixVideoTextDataset(Dataset):
    def __init__(self, root, split, seq_ids=None, frame_skip=1,
                 crop_size=224, pose_cache_dir=None):
        """If *seq_ids* is None load every row in <split>.corpus.csv
        Otherwise restrict to that subset (list of strings)."""
```

* **RGB pipeline**

  1. Load PNG frames from `fullFrame-256x256px`.
  2. Center‑crop to 224×224.
  3. `ToTensor` → `FloatTensor[T,3,224,224]`, normalise with ImageNet mean/std.
* **Pose pipeline** (Two‑Stream only)

  1. Query `keypoints.pkl` → `[T,133,3]`.
  2. Render per‑keypoint Gaussian (σ = 4) on 112×112.
  3. Max‑pool left vs right keypoint groups → **2‑channel heat‑map** `[T,2,112,112]`.
  4. Bilinear upscale to 224×224, same center crop.
  5. Cache to `keypoints/heatmaps/<seq>.npy` after first render.

**Sliding‑window rule** (both models)

* If `T > 200` frames, split into overlapping windows of 200 (stride = 100).  Per‑frame logits are averaged when windows overlap.

Batch dict returned:

```python
{
  "rgb":   FloatTensor [T,3,224,224],
  "pose":  FloatTensor [T,2,224,224] | None,
  "gloss": LongTensor [L]            | empty for test,
  "length": int                      # pre‑crop frame count
}
```

---

## 5  Grad‑CAM Utility ― `cam_runner.py`

* Uses **pytorch‑grad‑cam 1.5.2**.
* Runs **layer‑by‑layer** to minimise VRAM (`batch_size=1` in CAM lib).
* Forward pass under `torch.cuda.amp.autocast()`; backward in FP32.
* **Target score** default: max logit per frame.

  * If `--target-from-gt` flag is passed and `gloss` is non‑empty, back‑prop from *ground‑truth gloss ID* of the *median frame*.
* For 3‑D activations CAM returns `[T,H,W]`; overlay done frame‑wise on original RGB.

---

## 6  CLI ― `main.py`

```bash
python code/main.py \
    --seq 01April_2010_Thursday_heute_default-8 \
    --split train \
    --models slowfast twostream \
    --slowfast-ckpt checkpoints/slowfast_phoenix2014_dev_18.01_test_18.28.pt \
    --twostream-ckpt checkpoints/twostreamslr.ckpt \
    --target-from-gt \
    --device cuda:0
```

| flag               | description                       | default    |
| ------------------ | --------------------------------- | ---------- |
| `--seq`            | sequence ID or comma‑list \[,…]   | *all rows* |
| `--split`          | `train / dev / test`              | `dev`      |
| `--models`         | subset of `{slowfast, twostream}` | `slowfast` |
| `--target-from-gt` | use ground‑truth token for CAM    | `False`    |
| `--device`         | CUDA device string                | `cuda:0`   |

Outputs land in:

```
outputs/<seq>/
    slowfast/<layer>/frame_%03d.png
    twostream/<layer>/frame_%03d.png
    comparison.mp4                  # side‑by‑side 2×5 grid video
    preds.txt                       # GT, SlowFast, TwoStream strings
```

---

## 7  Helper: pre‑cache pose heat‑maps

```bash
python code/cache_pose_heatmaps.py --split train --seq 01April_2010_Thursday_heute_default-8 --device cuda:0
```

---

## 8  Test Matrix (must pass on first run)

| GPU  | cmd                                    | expected          | VRAM peak |
| ---- | -------------------------------------- | ----------------- | --------- |
| A40  | single `--seq` 200 f clip, both models | completes < 60 s  | ≤ 14 GB   |
| A100 | `--seq` list of 3 clips                | completes < 3 min | ≤ 18 GB   |

---

## 9  Implementation Notes for Codex

* **Do not refactor** checkpoints – load with `torch.load(..., map_location)`.
* **Keep** BN / Dropout layers in `eval()`.
* **Avoid** scatter‑gather gradients; only `.backward()` once per CAM.
* **Release hooks** after every CAM call ⇒ memory stable.
* **Set** `export TORCH_HOME=$PWD/checkpoints` if torchvision attempts URL fetch.

---

## 10  Acknowledgements / Citations

* SlowFast‑Sign: Ahn et al., *ICCV 2023*.
* TwoStream‑SLR: Bi et al., *ECCV 2022*.
* Grad‑CAM: Selvaraju et al., *ICCV 2017*.
* COCO WholeBody keypoints: Li et al., *ECCV 2020*.

---

> **End of spec** – code generated from this document **must run without manual edits** under the directory structure and environment described above.
