# code/cam_runner.py
from __future__ import annotations
from typing import List, Optional

import cv2
import numpy as np
import torch
from torch import nn
from pytorch_grad_cam import GradCAM


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

        # ---- cuDNN LSTM backward guard ----------------------
        saved_mode  = self.model.training
        saved_cudnn = torch.backends.cudnn.enabled
        torch.backends.cudnn.enabled = False
        self.model.train()

        try:
            for layer in self.layers:
                # context-manager handles hook cleanup
                with GradCAM(model=self.model, target_layers=[layer]) as cam:
                    with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
                        output = (
                            self.model(*inputs) if isinstance(inputs, list)
                            else self.model(inputs)
                        )                             # (B,C) or (B,T,C)

                    # choose default class if none provided
                    if class_id is None:
                        class_id = int(
                            output.mean(1).argmax() if output.ndim == 3 else output.argmax()
                        )

                    # ------------- spatial CAM ----------------
                    try:
                        targets = [SeqClassifierTarget(class_id)]
                        gcam = cam(input_tensor=input_tensor, targets=targets)
                        cams.append(gcam[0])              # 2-D / 3-D map
                        continue                          # success ⇒ next layer
                    except ValueError:
                        pass                              # fall through to 1-D

                    # ------------- temporal CAM --------------
                    # run backward manually for 1-D tensors
                    if output.ndim != 3:
                        print(f"[CAM] Skipped {layer} – unsupported shape {tuple(output.shape)}")
                        continue

                    # loss = output[:, :, class_id].mean()
                    # self.model.zero_grad()
                    loss = output[:, :, class_id].mean()
                    self.model.zero_grad()
                    output.retain_grad()          # ← NEW (capture grads on non-leaf)
                    loss.backward(retain_graph=True)
                    cam_1d = time_gradcam(output.detach(), output.grad)
                    cams.append(cam_1d)                  # (T,) curve
                    # cams list now holds either:
                    #   * 3-D array (T,H,W)  – spatial map
                    #   * 1-D array (T,)     – temporal curve


        finally:
            if not saved_mode:
                self.model.eval()
            torch.backends.cudnn.enabled = saved_cudnn

        return cams


# ────────────────────────────────────────────────────────────
def overlay_heatmap(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    Blend CAM mask onto RGB image.

    • If mask is 2-D or 3-D  → classic heat-map overlay.
    • If mask is 1-D        → draw a coloured bar at the bottom of the frame.
    """
    if mask.ndim == 1:                                   # 1-D temporal curve
        # normalise & convert to RGB bar (width = image width, height = 8 px)
        bar_h = 8
        curve = (mask / (mask.max() + 1e-8) * 255).astype(np.uint8)
        bar   = cv2.applyColorMap(curve, cv2.COLORMAP_JET)      # (T,3)
        bar   = cv2.resize(bar, (image.shape[1], bar_h), interpolation=cv2.INTER_NEAREST)
        # stack bar under the frame
        combined = np.vstack([image, bar])
        return combined

    # -------- spatial mask --------
    heat = cv2.applyColorMap((mask * 255).astype(np.uint8), cv2.COLORMAP_JET)
    if image.max() > 1:
        image = image.astype(np.float32) / 255.0
    blended = np.clip(0.5 * heat.astype(float) + 0.5 * image.astype(float), 0, 255)
    return blended.astype(np.uint8)
