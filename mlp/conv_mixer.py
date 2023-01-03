import torch
import torchsummary
from torch import nn


class ConvMixer(nn.Module):

    def __init__(
        self,
        in_channels: int,
        hidden_dim: int,
        n_layers: int,
        kernel_size: int = 5,
        patch_size: int = 2,
        num_classes: int = 10
    ):
        super().__init__()

        sublayer = nn.Sequential(
            nn.Conv2d(
                hidden_dim,
                hidden_dim,
                kernel_size,
                groups=hidden_dim,
                padding='same'
            ),
            nn.GELU(),
            nn.BatchNorm2d(hidden_dim)
        )

        self.convmixer_layers = nn.Sequential(
            nn.Conv2d(
                in_channels,
                hidden_dim,
                kernel_size=patch_size,
                stride=patch_size
            ),
            nn.GELU(),
            nn.BatchNorm2d(hidden_dim),
            *[
                nn.Sequential(
                    Residual(sublayer),
                    nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1),
                    nn.GELU(),
                    nn.BatchNorm2d(hidden_dim)
                ) for _ in range(n_layers)
            ],
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
        )
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):

        x = self.convmixer_layers(x)
        out = self.classifier(x)

        return out


class Residual(nn.Module):
    def __init__(self, layer):
        super().__init__()
        self.layer = layer

    def forward(self, x):
        return self.layer(x) + x


if __name__ == '__main__':
    import torchsummary

    # test code
    convmixer = ConvMixer(
        in_channels=3,
        hidden_dim=256,
        n_layers=8,
        kernel_size=12,
        patch_size=4,
        num_classes=10
    )

    torchsummary.summary(convmixer.cuda(), (3, 32, 32))
