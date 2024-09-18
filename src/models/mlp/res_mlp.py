import torch
import numpy as np
from torch import nn
from einops.layers.torch import Rearrange


class AffineTransformLayer(nn.Module):
    def __init__(
        self, 
        hidden_dim: int
    ) -> None:
        """
        Args
            hidden_dim : int
                Dimensionality of AffineTransform Layer.
        """
        super(self, AffineTransformLayer).__init__()

        self.alpha = nn.Parameter(torch.ones([1, 1, hidden_dim]))
        self.beta = nn.Parameter(torch.zeros([1, 1, hidden_dim]))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.alpha * x + self.beta


class FeedForwardLayer(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        dropout_p: float = 0.
    ) -> None:
        super(FeedForwardLayer, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout_p),
            nn.Linear(hidden_dim, input_dim),
            nn.Dropout(dropout_p)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class CommunicationLayer(nn.Module):

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_patch: int,
        dropout: float,
        init_values: float = 1e-4
    ) -> None:
        super(CommunicationLayer, self).__init__()

        self.pre_affine = AffineTransformLayer(input_dim)
        self.token_mix = nn.Sequential(
            Rearrange('b n d -> b d n'),
            nn.Linear(num_patch, num_patch),
            Rearrange('b d n -> b n d'),
        )
        self.ff = nn.Sequential(
            FeedForwardLayer(input_dim, hidden_dim, dropout),
        )
        self.post_affine = AffineTransformLayer(input_dim)
        self.gamma_1 = nn.Parameter(
            init_values * torch.ones((input_dim)),
            requires_grad=True
        )
        self.gamma_2 = nn.Parameter(
            init_values * torch.ones((input_dim)),
            requires_grad=True
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pre_affine(x)
        x = x + self.gamma_1 * self.token_mix(x)
        x = self.post_affine(x)
        x = x + self.gamma_2 * self.ff(x)
        return x


class ResMLP(nn.Module):

    def __init__(
        self,
        img_size: int,
        in_channels: int,
        input_dim: int,
        patch_size: int,
        hidden_dim: int,
        dropout_p: float,
        n_layers: int,
        num_classes: int,
    ) -> None:
        super(ResMLP, self).__init__()

        assert img_size % patch_size == 0, "`img_size` must be divisible by the `patch_size`."
        self.num_patch = (img_size // patch_size) ** 2
        self.to_patch_embedding = nn.Sequential(
            nn.Conv2d(in_channels, input_dim, patch_size, patch_size),
            Rearrange('b c h w -> b (h w) c'),
        )
        self.feedforward_layers = nn.Sequential(
            *[
                CommunicationLayer(
                    input_dim,
                    hidden_dim,
                    self.num_patch,
                    dropout_p,
                ) for _ in range(n_layers)
            ]
        )
        self.affine = AffineTransformLayer(input_dim)
        self.classifier = nn.Linear(input_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.to_patch_embedding(x)
        x = self.feedforward_layers(x)
        x = self.affine(x)
        x = x.mean(dim=1)
        return self.classifier(x)


if __name__ == '__main__':
    import torchsummary

    # test code
    resmlp = ResMLP(
        img_size=32,
        in_channels=3,
        patch_size=4,
        input_dim=128,
        hidden_dim=512,
        dropout_p=0.,
        n_layers=8,
        num_classes=10,
    )

    data = torch.randn(size=(3, 32, 32))
    torchsummary.summary(resmlp.cuda(), (3, 32, 32))