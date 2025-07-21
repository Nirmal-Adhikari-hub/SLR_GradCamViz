from __future__ import annotations

from typing import Iterable, Optional, List

import cv2
import numpy as np
import torch
from torch import nn
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget


class CAMRunner:
    '''Run Grad-CAM layer-by-layer using minimal VRAM'''

    def __init__(self, model: nn.Module, target_layers: List[nn.Module]):
        self.model = model
        self.layers = target_layers


    def run(self, 
            inputs: torch.Tensor | List[torch.Tensor],
            device: torch.device,
            class_id: Optional[int] =None,
            ) -> List[np.ndarray]:
        """
            Run Grad-CAM visualization on the given inputs across all target layers.

            This method generates Class Activation Maps using the Gradient-weighted Class Activation
            Mapping (Grad-CAM) technique for the provided input(s). It processes each configured target
            layer and returns the resulting heatmaps as grayscale numpy arrays.

            Args:
                inputs (torch.Tensor | List[torch.Tensor]): Input tensor(s) to run through the model.
                    Can be a single tensor or a list of tensors for models with multiple inputs.
                device (torch.device): Device to run the computation on (CPU or CUDA).
                class_id (Optional[int]): The class index to generate Grad-CAM for. If None,
                    the predicted class with the highest score will be used.

            Returns:
                List[np.ndarray]: A list of grayscale Grad-CAM visualizations as numpy arrays,
                    one for each target layer.
        """
        if isinstance(inputs, (list, tuple)):
            inputs = [x.to(device) for x in inputs]
        else:
            inputs = inputs.to(device)

        results: List[np.ndarray] = []
        for layer in self.layers:
            cam = GradCAM(model=self.model, target_layers=[layer])
            with torch.cuda.amp.autocast(enabled=device.type == 'cuda'):
                outputs = self.model(*inputs) if isinstance(inputs, list) else self.model(inputs)
            if class_id is None:
                if outputs.ndim == 3:
                    class_id = int(outputs[0, outputs.shape[1] // 2].argmax())
                else:
                    class_id = int(outputs[0].argmax())
            targets = [ClassifierOutputTarget(class_id)]

            # grayscale_cam = cam(input_tensor=inputs, targets=targets)
            input_tensor = inputs[0] if isinstance(inputs, list) else inputs
            grayscale_cam = cam(input_tensor=input_tensor, targets=targets)
            cam.clear_hooks()
            results.append(grayscale_cam[0])
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