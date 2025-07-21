from __future__ import annotations

import torch
from torch import nn


class SlowFastSign(nn.Module):
    """Toy SlowFast-Sign wrapper exposing CAM target layers. """

    def __init__(self, num_classes: int = 1296):
        super().__init__()
        self.conv2d = nn.Module()
        self.conv2d.slow_path = nn.Sequential(
            nn.Conv3d(3, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv3d(8, 16, kernel_size=3, padding=1),
            nn.ReLU(),
        )

        self.conv2d.fast_path = nn.Sequential(
            nn.Conv3d(3, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv3d(8, 8, kernel_size=3, padding=1),
            nn.ReLU(),
        )

        self.conv1d = nn.Module()
        self.conv1d.fused_features = nn.Conv1d(24,32, kernel_size=1)

        self.temporal_model = nn.ModuleList(
            [nn.LSTM(32, 32, batch_first=True) for _ in range(3)]
        )

        self.classifier = nn.ModuleList(
            [nn.Linear(32, num_classes) for _ in range(3)]
        )


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        slow = self.conv2d.slow_path(x)
        fast = self.conv2d.fast_path(x)

        fuse = torch.cat([
            slow.mean((3,4)),
            nn.functional.interpolate(fast, size=slow.shape[2:], mode="nearest").mean((3,4)),
        ], dim=1)

        fuse = self.conv1d.fused_features(fuse)
        out, _ = self.temporal_model[1](fuse.transpose(1,2))
        logits = self.classifier[0](out)
        return logits
    
    @property
    def target_layers(self):
        return[
            self.conv2d.slow_path,
            self.conv2d.fast_path,
            self.conv2d.fast_path[-1],
            self.conv1d.fused_features,
            # self.temporal_model[1]
        ]