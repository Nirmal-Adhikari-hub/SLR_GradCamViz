# cam_runner.py
from __future__ import annotations
from typing import List, Optional

import cv2
import numpy as np
import torch
from torch import nn
from pytorch_grad_cam import GradCAM


# ------------------------------------------------------------------
# Simple scalar target: average selected class logit over time
# ------------------------------------------------------------------
class SeqClassifierTarget:
    def __init__(self, class_idx: int):
        self.class_idx = class_idx

    def __call__(self, model_out: torch.Tensor) -> torch.Tensor:
        """
        model_out shape:
            (B, C)       – return mean over batch of class logit
            (B, T, C)    – return mean over batch & time of class logit
        """
        if model_out.ndim == 2:
            return model_out[:, self.class_idx].mean()
        elif model_out.ndim == 3:
            return model_out[:, :, self.class_idx].mean()
        raise ValueError(f"Unexpected output shape {model_out.shape}")


# ------------------------------------------------------------------
class CAMRunner:
    """Generate Grad-CAM heat-maps for a list of layers."""

    def __init__(self, model: nn.Module, target_layers: List[nn.Module]):
        self.model = model
        self.layers = target_layers

    def run(
        self,
        inputs: torch.Tensor | List[torch.Tensor],
        device: torch.device,
        class_id: Optional[int] = None,
    ) -> List[np.ndarray]:
        """
        Args
        ----
        inputs   : (B,C,T,H,W) Tensor **or** list [rgb, pose]
        device   : torch.device
        class_id : gloss ID to visualise; if None uses model arg-max.

        Returns
        -------
        List[np.ndarray]  – one CAM per target layer (T,H,W or H,W).
        """
        # Move tensors to device & pick representative tensor for Grad-CAM
        if isinstance(inputs, (list, tuple)):
            inputs = [x.to(device) for x in inputs]
            input_tensor = inputs[0]
        else:
            inputs = inputs.to(device)
            input_tensor = inputs

        cams: List[np.ndarray] = []

        # ---- cuDNN LSTM backward guard -----------------------------------
        saved_mode  = self.model.training
        saved_cudnn = torch.backends.cudnn.enabled
        torch.backends.cudnn.enabled = False
        self.model.train()
        # ------------------------------------------------------------------

        try:
            for layer in self.layers:
                with GradCAM(model=self.model, target_layers=[layer]) as cam:
                    # forward
                    with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
                        out = self.model(*inputs) if isinstance(inputs, list) else self.model(inputs)

                    # choose default class if not specified
                    if class_id is None:
                        class_id = int(out.mean(1).argmax()) if out.ndim == 3 else int(out.argmax())

                    targets = [SeqClassifierTarget(class_id)]
                    grayscale_cam = cam(input_tensor=input_tensor, targets=targets)
                    cams.append(grayscale_cam[0])          # batch index 0

        finally:  # always restore state
            if not saved_mode:
                self.model.eval()
            torch.backends.cudnn.enabled = saved_cudnn

        return cams


# ------------------------------------------------------------------
def overlay_heatmap(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Blend heatmap onto image and return uint8 RGB."""
    heat = cv2.applyColorMap((mask * 255).astype(np.uint8), cv2.COLORMAP_JET)
    out  = np.clip(0.5 * heat.astype(float) + 0.5 * image.astype(float), 0, 255)
    return out.astype(np.uint8)
