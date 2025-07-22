# code/cam_runner.py
from __future__ import annotations
from typing import List, Optional

import cv2
import numpy as np
import torch
from torch import nn
from pytorch_grad_cam import GradCAM

from pathlib import Path


# ────────────────────────────────────────────────────────────
# 1-D Grad-CAM helper for (B, T, C) tensors
# ────────────────────────────────────────────────────────────
def time_gradcam(features: torch.Tensor, grads: torch.Tensor) -> np.ndarray:
    """
    features : (B, T, C)
    grads    : (B, T, C)
    returns  : (T,) normalised importance per frame
    """
    # channel-wise weights
    weights = grads.mean(dim=(0, 2))                 # (T,)
    # energy over channels
    energy  = features.pow(2).sum(-1).sqrt().mean(0) # (T,)
    cam     = torch.relu(weights * energy)
    cam     = cam / (cam.max() + 1e-8)
    return cam.detach().cpu().numpy()                # (T,)


# ────────────────────────────────────────────────────────────
# Scalar target for Grad-CAM on 4-D / 5-D tensors
# ────────────────────────────────────────────────────────────
class SeqClassifierTarget:
    def __init__(self, class_idx: int):
        self.class_idx = class_idx

    def __call__(self, output: torch.Tensor) -> torch.Tensor:
        # (B,C) or (B,T,C) → scalar
        if output.ndim == 2:
            return output[:, self.class_idx].mean()
        elif output.ndim == 3:
            return output[:, :, self.class_idx].mean()
        raise ValueError(f"Unexpected output shape {output.shape}")


# ────────────────────────────────────────────────────────────
# Main runner
# ────────────────────────────────────────────────────────────
class CAMRunner:
    """Generates a list of CAMs per target layer (2-D or 1-D)."""

    def __init__(self, model: nn.Module, target_layers: List[nn.Module]):
        self.model = model
        self.layers = target_layers
        

    def run(
        self,
        inputs: torch.Tensor | List[torch.Tensor],
        device: torch.device,
        class_id: Optional[int] = None,
    ) -> List[np.ndarray]:
        # ------------- move tensors to device ----------------
        if isinstance(inputs, (list, tuple)):
            inputs = [x.to(device) for x in inputs]
            input_tensor = inputs[0]
        else:
            inputs = inputs.to(device)
            input_tensor = inputs

        cams: List[np.ndarray] = []

        saved_mode = self.model.training          # keep original mode
        self.model.train()                        # RNNs need train mode

        try:
            for layer in self.layers:
                layer_saved_cudnn = torch.backends.cudnn.enabled  # conv3d flag

                with GradCAM(model=self.model, target_layers=[layer]) as cam:
                    with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
                        output = (
                            self.model(*inputs) if isinstance(inputs, list)
                            else self.model(inputs)
                        )  # (B,C) or (B,T,C)

                    if class_id is None:
                        class_id = int(
                            output.mean(1).argmax() if output.ndim == 3 else output.argmax()
                        )

                    # ---------- spatial CAM (H,W or T,H,W) ----------
                    try:
                        targets = [SeqClassifierTarget(class_id)]
                        gcam = cam(input_tensor=input_tensor, targets=targets)
                        cams.append(gcam[0])
                    except ValueError:
                        # ---------- temporal CAM (T,) ----------
                        if output.ndim != 3:
                            print(f"[CAM] Skipped {layer} – shape {tuple(output.shape)}")
                        else:
                            torch.backends.cudnn.enabled = False  # disable for LSTM backward
                            loss = output[:, :, class_id].mean()
                            self.model.zero_grad()
                            output.retain_grad()
                            loss.backward(retain_graph=True)
                            cam_1d = time_gradcam(output.detach(), output.grad)
                            cams.append(cam_1d)

                # restore cuDNN flag before next layer
                torch.backends.cudnn.enabled = layer_saved_cudnn

        finally:
            if not saved_mode:
                self.model.eval()

        return cams


# ────────────────────────────────────────────────────────────
def overlay_heatmap(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    Blend CAM mask onto RGB image.

    • mask 2-D : classic heat-map overlay.
    • mask 1-D : draw a coloured bar under the frame.
    """
    # esnure both inputs are numpy arrays
    if torch.is_tensor(mask):
        mask = mask.detach().cpu().numpy()
    mask = np.asarray(mask, dtype=np.float32)
    if torch.is_tensor(image):
        image = image.detach().cpu().numpy()
    image = np.asarray(image, dtype=np.float32)

    if mask.ndim == 1:                                   # temporal curve
        bar_h  = 8
        curve  = (mask / (mask.max() + 1e-8) * 255).astype(np.uint8)  # (T,)
        bar    = cv2.applyColorMap(curve, cv2.COLORMAP_JET)           # (T,3)
        bar    = cv2.resize(bar, (image.shape[1], bar_h),
                            interpolation=cv2.INTER_NEAREST)
        return np.vstack([image, bar])                   # stack under frame

    # ─ spatial mask ─
    mask = np.squeeze(mask, axis=-1) if mask.ndim == 3 else mask  # (H,W)
    heat = cv2.applyColorMap(
        np.uint8(np.clip(mask * 255, 0, 255)), cv2.COLORMAP_JET
    )
    blended = np.clip(0.5 * heat.astype(float) +
                      0.5 * image.astype(float), 0, 255)
    return blended.astype(np.uint8)


# ------------------------------------------------------------------
def save_or_overlay(
    frame: np.ndarray,
    cam:   np.ndarray,          # single 2-D slice  OR  1-D curve
    layer_dir: Path,
    t_idx: int
):
    """
    * 2-D CAM  → overlay and save PNG for this frame.
    * 1-D CAM  → save numpy once (t_idx==0).
    """
    if cam.ndim == 1:                                  # (T,)
        if t_idx == 0:                                 # save only once
            np.save(layer_dir / "temporal_cam.npy", cam)
        return

    out = overlay_heatmap(frame, cam)
    cv2.imwrite(str(layer_dir / f"frame_{t_idx:03d}.png"), out)
