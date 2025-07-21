from __future__ import annotations

from typing import Iterable, Optional, List

import cv2
import numpy as np
import torch
from torch import nn
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget


class CAMRunner:
    """Run Grad-CAM on 3-D video models with minimal VRAM."""

    class TemporalTarget:           # <-- new helper
        def __init__(self, time_idx: int, class_id: int):
            self.t = time_idx
            self.c = class_id
        def __call__(self, model_out):
            # model_out shape (B, T, C) → scalar
            return model_out[:, self.t, self.c]

    def __init__(self, model: nn.Module, target_layers: List[nn.Module]):
        self.model = model
        self.layers = target_layers

    def run(
        self,
        inputs: torch.Tensor | List[torch.Tensor],
        device: torch.device,
        class_id: Optional[int] = None,
    ) -> List[np.ndarray]:

        if isinstance(inputs, (list, tuple)):
            inputs = [x.to(device) for x in inputs]
            input_tensor = inputs[0]          # Grad-CAM only needs one
        else:
            inputs = inputs.to(device)
            input_tensor = inputs

        # forward once per layer (saves RAM)
        with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
            outputs = (
                self.model(*inputs) if isinstance(inputs, list) else self.model(inputs)
            )  # (1, T, C)

        # choose default class_id if not provided
        if class_id is None:
            mid = outputs.shape[1] // 2
            class_id = int(outputs[0, mid].argmax())

        time_idx = outputs.shape[1] // 2  # middle frame
        target = [self.TemporalTarget(time_idx, class_id)]

        results: List[np.ndarray] = []
        for layer in self.layers:
            cam = GradCAM(model=self.model, target_layers=[layer])
            grayscale_cam = cam(input_tensor=input_tensor, targets=target)
            cam.clear_hooks()
            results.append(grayscale_cam[0])        # (T,H,W) or (H,W)
        return results

    

def overlay_heatmap(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    Overlay a heatmap on an image.
    
    Args:
        image (np.ndarray): The original image.
        mask (np.ndarray): The heatmap to overlay, should be the same size as the image.
    
    Returns:
        np.ndarray: The image with the heatmap overlayed.
    """
    heatmap = cv2.applyColorMap(np.uint8(255*mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    if image.max() > 1:
        image = image / 255.0
    overlay = heatmap * 0.5 + image * 0.5 # Control the intensity of the overlay (OPACITY)
    overlay = np.clip(overlay, 0, 1)
    return np.uint8(overlay * 255)