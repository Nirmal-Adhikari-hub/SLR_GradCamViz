from __future__ import annotations

import torch
from torch import nn

class TwoStreamSLR(nn.Module):
    """Toy TwoStream network for CAM vizs"""

    def __init__(self, num_classes: int = 1296):
        super().__init__()
        self.rgb_stream = nn.Sequential(
            nn.Conv3d(3, 16, 3, padding=1),
            nn.ReLU(),
            nn.Conv3d(16, 32, 3, padding=1),
            nn.ReLU(),
        )

        self.pose_stream = nn.Sequential(
            nn.Conv3d(2, 16, 3, padding=1),
            nn.ReLU(),
            nn.Conv3d(16, 32, 3, padding=1),
            nn.ReLU(),
        )

        self.fusion_module = nn.Sequential(
            nn.Conv3d(64, 64, kernel_size=1),
            nn.ReLU(),
        )

        self.visual_head = nn.Linear(64, num_classes)


    def forward(self, rgb: torch.Tensor, pose: torch.Tensor) -> torch.Tensor:
        r = self.rgb_stream(rgb)
        p = self.pose_stream(pose)

        feat = torch.cat([r, p], dim=1)
        feat = self.fusion_module(feat)
        feat = feat.mean((2,3,4))
        logits = self.visual_head(feat)
        return logits
    

    @property
    def target_layers(self):
        return [
            self.rgb_stream[0],
            self.pose_stream[0],
            self.fusion_module[0],
            self.rgb_stream[-1],
            # self.visual_head,
        ]