import torch.nn as nn
import torch
class Discriminator3D(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            # Layer 1
            nn.Conv3d(1, 64, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            # Layer 2
            nn.Conv3d(64, 128, 4, stride=2, padding=1),
            nn.InstanceNorm3d(128),
            nn.LeakyReLU(0.2, inplace=True),
            # Layer 3
            nn.Conv3d(128, 256, 4, stride=2, padding=1),
            nn.InstanceNorm3d(256),
            nn.LeakyReLU(0.2, inplace=True),
            # Layer 4
            nn.Conv3d(256, 512, 4, stride=2, padding=1),
            nn.InstanceNorm3d(512),
            nn.LeakyReLU(0.2, inplace=True),
            # Final Flatten and Dense
            nn.AdaptiveAvgPool3d(1),
            nn.Flatten(),
            nn.Linear(512, 1),
            # No Sigmoid here if using BCEWithLogitsLoss
        )

    def forward(self, x):
        return self.model(x)

