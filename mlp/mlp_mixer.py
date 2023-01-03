import torch
from torch import nn
from einops.layers.torch import Rearrange


class MlpMixer(nn.Module):
    """MLP-Mixer network.
    Args
    ----
        image_size : int
            Height and width (assuming it is a square) of the input image.
        patch_size : int
            Height and width (assuming it is a square) of the patches. Note
            that we assume that `image_size % patch_size == 0`.
        tokens_mlp_dim : int
            Hidden dimension for the `MlpBlock` when doing the token mixing.
        channels_mlp_dim : int
            Hidden dimension for the `MlpBlock` when diong the channel mixing.
        n_classes : int
            Number of classes for classification.
        hidden_dim : int
            Dimensionality of patch embeddings.
        n_layers : int
            The number of `MixerBlock`s in the architecture.

    Attributes
    ----------
        patch_embedder : nn.Conv2D
            Splits the image up into multiple patches and then embeds each of them
            (using shared weights).
        blocks : nn.ModuleList
            List of `MixerBlock` instances.
        pre_head_norm : nn.LayerNorm
            Layer normalization applied just before the classification head.
        head_classifier : nn.Linear
            The classification head.
    """

    def __init__(
        self,
        in_channels: int,
        image_size: int,
        patch_size: int,
        tokens_mlp_dim: int,
        channels_mlp_dim: int,
        hidden_dim: int,
        dropout_p: float,
        n_layers: int,
        num_classes: int,
    ):
        super().__init__()
        n_patches = (image_size // patch_size) ** 2

        self.patch_embedder = nn.Sequential(
            nn.Conv2d(
                in_channels,
                hidden_dim,
                kernel_size=patch_size,
                stride=patch_size,
            ),
            Rearrange('n c h w -> n (h w) c')
        )

        self.layers = nn.Sequential(
            *[
                MixerBlock(
                    n_patches=n_patches,
                    hidden_dim=hidden_dim,
                    tokens_mlp_dim=tokens_mlp_dim,
                    channels_mlp_dim=channels_mlp_dim,
                    dropout_p=dropout_p
                )
                for _ in range(n_layers)
            ],
            nn.LayerNorm(hidden_dim)
        )

        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        """Run the forward pass.
        Args
        ----
        x : torch.Tensor
            Input batch of square images of shape
            `(batch_size, n_channels, image_size, image_size)`.

        Returns
        -------
        torch.Tensor
            Class logits of shape `(batch_size, n_classes)`.
        """

        x = self.patch_embedder(x)  # (batch_size, n_patches, hidden_dim)
        x = self.layers(x)          # (batch_size, n_patches, hidden_dim)
        x = x.mean(dim=1)           # (batch_size, hidden_dim)
        out = self.classifier(x)    # (batch_size, n_classes)

        return out


class FeedforwardLayer(nn.Module):
    """Multilayer perceptron.
    Args
    ----
        dim : int
            Input and output dimension of the entire block. Inside of the mixer
            it will either be equal to `n_patches` or `hidden_dim`.
        mlp_dim : int
            Dimension of the hidden layer.

    Attributes
    ----------
        linear_1, linear_2 : nn.Linear
            Linear layers.
        activation : nn.GELU
            Activation.
    """

    def __init__(self, dim, mlp_dim=None, dropout_p: float = 0.):
        super().__init__()

        mlp_dim = dim if mlp_dim is None else mlp_dim
        self.linear_1 = nn.Linear(dim, mlp_dim)
        self.drop1 = nn.Dropout(dropout_p)
        self.activation = nn.GELU()
        self.drop2 = nn.Dropout(dropout_p)
        self.linear_2 = nn.Linear(mlp_dim, dim)

    def forward(self, x):
        """Run the forward pass.
        Args
        ----
            x : torch.Tensor
                Input tensor of shape `(batch_size, n_channels, n_patches)` or
                `(batch_size, n_patches, n_channels)`.

        Returns
        -------
            torch.Tensor
                Output tensor that has exactly the same shape as the input `x`.
        """
        x = self.linear_1(x)  # (batch_size, *, mlp_dim)
        x = self.activation(x)  # (batch_size, *, mlp_dim)
        x = self.drop1(x)
        x = self.linear_2(x)  # (batch_size, *, dim)
        x = self.drop2(x)

        return x


class MixerBlock(nn.Module):
    """Mixer block that contains two `MlpBlock`s and two `LayerNorm`s.
    Args
    ----
        n_patches : int
            Number of patches the image is split up into.
        hidden_dim : int
            Dimensionality of patch embeddings.
        tokens_mlp_dim : int
            Hidden dimension for the `MlpBlock` when doing token mixing.
        channels_mlp_dim : int
            Hidden dimension for the `MlpBlock` when doing channel mixing.

    Attributes
    ----------
        norm_1, norm_2 : nn.LayerNorm
            Layer normalization.
        token_mlp_block : MlpBlock
            Token mixing MLP.
        channel_mlp_block : MlpBlock
            Channel mixing MLP.
    """

    def __init__(
        self,
        n_patches: int,
        hidden_dim: int,
        tokens_mlp_dim: int,
        channels_mlp_dim: int,
        dropout_p: float
    ):
        super().__init__()

        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.token_mlp_block = FeedforwardLayer(
            n_patches, tokens_mlp_dim, dropout_p
        )
        self.channel_mlp_block = FeedforwardLayer(
            hidden_dim, channels_mlp_dim, dropout_p
        )

    def forward(self, x):
        """
        Args
            x : torch.Tensor
                Tensor of shape `(batch_size, n_patches, hidden_dim)`.

        Returns
            torch.Tensor
                Tensor of the same shape as `x`, i.e.
                `(batch_size, n_patches, hidden_dim)`.
        """

        y = self.norm1(x)       # (batch_size, n_patches, hidden_dim)
        y = y.permute(0, 2, 1)  # (batch_size, hidden_dim, n_patches)
        y = self.token_mlp_block(y)  # (batch_size, hidden_dim, n_patches)
        y = y.permute(0, 2, 1)       # (batch_size, n_patches, hidden_dim)
        x = x + y           # (batch_size, n_patches, hidden_dim)
        y = self.norm2(x)   # (batch_size, n_patches, hidden_dim)

        # (batch_size, n_patches, hidden_dim)
        res = x + self.channel_mlp_block(y)

        return res


if __name__ == '__main__':
    from torchsummary import summary

    mlpmixer = MlpMixer(
        in_channels=3,
        image_size=32,
        patch_size=4,
        tokens_mlp_dim=64,
        channels_mlp_dim=512,
        hidden_dim=128,
        dropout_p=0.,
        n_layers=8,
        num_classes=10,
    )
    print(mlpmixer)
    summary(mlpmixer.cuda(), (3, 32, 32))
