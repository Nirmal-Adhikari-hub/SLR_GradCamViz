from __future__ import annotations
from typing import List, Optional

import cv2
import numpy as np
import torch
from torch import nn
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import BaseCAMTarget


# ------------------------------------------------------------------
# Scalar target: average logit of class_idx over time (if any)
# ------------------------------------------------------------------
class SeqClassifierTarget(BaseCAMTarget):
    def __init__(self, class_idx: int):
        self.class_idx = class_idx

    def __call__(self, model_out: torch.Tensor) -> torch.Tensor:
        # (B,C) or (B,T,C)  ➜  scalar
        if model_out.ndim == 2:               # (B, C)
            return model_out[:, self.class_idx].mean()
        elif model_out.ndim == 3:             # (B, T, C)
            return model_out[:, :, self.class_idx].mean()
        else:
            raise ValueError(f"Unexpected output shape {tuple(model_out.shape)}")


# ------------------------------------------------------------------
# Grad-CAM runner
# ------------------------------------------------------------------
class CAMRunner:
    def __init__(self, model: nn.Module, target_layers: List[nn.Module]):
        self.model = model
        self.layers = target_layers

    def run(
        self,
        inputs: torch.Tensor | List[torch.Tensor],
        device: torch.device,
        class_id: Optional[int] = None,
    ) -> List[np.ndarray]:
        """Return list of CAMs, one per target layer."""
        if isinstance(inputs, (list, tuple)):
            inputs = [x.to(device) for x in inputs]
            input_tensor = inputs[0]          # Grad-CAM needs a representative tensor
        else:
            inputs = inputs.to(device)
            input_tensor = inputs

        cams: List[np.ndarray] = []
        for layer in self.layers:
            cam = GradCAM(model=self.model, target_layers=[layer])

            with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
                output = (
                    self.model(*inputs) if isinstance(inputs, list)
                    else self.model(inputs)
                )                            # (B,C) or (B,T,C)

            if class_id is None:
                argmax_c = (
                    output[:, :, :].mean(1).argmax() if output.ndim == 3
                    else output.argmax()
                )
                class_id = int(argmax_c)

            targets = [SeqClassifierTarget(class_id)]
            grayscale_cam = cam(input_tensor=input_tensor, targets=targets)
            cams.append(grayscale_cam[0])    # batch idx 0
            cam.clear_hooks()

        return cams


# ------------------------------------------------------------------
def overlay_heatmap(img: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Overlay CAM on RGB image (uint8)."""
    heat = cv2.applyColorMap((mask * 255).astype(np.uint8), cv2.COLORMAP_JET)
    blended = np.clip(0.5 * heat.astype(float) + 0.5 * img.astype(float), 0, 255)
    return blended.astype(np.uint8)
