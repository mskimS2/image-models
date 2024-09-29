import torch
import timm
from torch import nn
import numpy as np
from einops import repeat, rearrange
from einops.layers.torch import Rearrange
from timm.models.layers import trunc_normal_
from timm.models.vision_transformer import Block
from typing import Tuple, List
import pytorch_lightning as pl


def random_indexes(size: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate shuffled forward and backward indexes for a given size.
    Args:
    - size (int): The size of the index array.
    Returns:
    - Tuple[np.ndarray, np.ndarray]: Forward and backward indexes.
    """
    forward_indexes = np.arange(size)
    np.random.shuffle(forward_indexes)
    backward_indexes = np.argsort(forward_indexes)
    return forward_indexes, backward_indexes


def take_indexes(sequences: torch.Tensor, indexes: torch.Tensor) -> torch.Tensor:
    """
    Gather the sequences based on the given indexes.
    Args:
    - sequences (torch.Tensor): The input sequence tensor.
    - indexes (torch.Tensor): The indexes for gathering.
    Returns:
    - torch.Tensor: Gathered sequences based on the provided indexes.
    """
    device = sequences.device
    indexes = indexes.to(device)
    return torch.gather(sequences, 0, repeat(indexes, "t b -> t b c", c=sequences.shape[-1]))


class PatchShuffle(nn.Module):
    def __init__(self, ratio: float):
        """
        PatchShuffle shuffles the patches randomly based on a given ratio.
        Args:
        - ratio (float): The ratio of patches to shuffle.
        """
        super(PatchShuffle, self).__init__()
        self.ratio = ratio

    def forward(self, patches: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Shuffle patches and return shuffled patches along with forward and backward indexes.
        Args:
        - patches (torch.Tensor): Input patches tensor of shape (T, B, C).
        Returns:
        - Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: Shuffled patches, forward indexes, and backward indexes.
        """
        T, B, C = patches.shape
        remain_T = int(T * (1 - self.ratio))

        # Generate forward and backward indexes for each batch
        indexes = [random_indexes(T) for _ in range(B)]
        forward_indexes = torch.as_tensor(np.stack([i[0] for i in indexes], axis=-1), dtype=torch.long)
        backward_indexes = torch.as_tensor(np.stack([i[1] for i in indexes], axis=-1), dtype=torch.long)

        # Shuffle patches and keep only the remaining ones
        patches = take_indexes(patches, forward_indexes)
        patches = patches[:remain_T]

        return patches, forward_indexes.to(patches.device), backward_indexes.to(patches.device)


class MAEEncoder(torch.nn.Module):
    def __init__(
        self,
        img_size: int = 32,
        patch_size: int = 2,
        emb_dim: int = 192,
        num_layers: int = 12,
        num_head: int = 3,
        mask_ratio: float = 0.75,
    ):
        """
        Masked Autoencoder (MAE) Encoder module.
        Args:
        - img_size (int): Size of the input image (assumed square).
        - patch_size (int): Size of the patches.
        - emb_dim (int): Embedding dimension.
        - num_layers (int): Number of transformer layers.
        - num_heads (int): Number of attention heads.
        - mask_ratio (float): Ratio of patches to be masked.
        """
        super(MAEEncoder, self).__init__()

        self.cls_token = torch.nn.Parameter(torch.zeros(1, 1, emb_dim))
        self.pos_embedding = torch.nn.Parameter(torch.zeros((img_size // patch_size) ** 2, 1, emb_dim))
        self.shuffle = PatchShuffle(mask_ratio)
        self.patchify = torch.nn.Conv2d(3, emb_dim, patch_size, patch_size)
        self.transformer = torch.nn.Sequential(*[Block(emb_dim, num_head) for _ in range(num_layers)])
        self.layer_norm = torch.nn.LayerNorm(emb_dim)
        self._init_weight()

    def _init_weight(self):
        """Initialize weights for the class token and position embedding."""
        trunc_normal_(self.cls_token, std=0.02)
        trunc_normal_(self.pos_embedding, std=0.02)

    def forward(self, image: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the encoder.
        Args:
        - image (torch.Tensor): Input image tensor of shape (B, 3, H, W).
        Returns:
        - Tuple[torch.Tensor, torch.Tensor]: Encoded features and backward indexes.
        """
        # Convert image to patches and add position embedding
        patches = self.patchify(image)
        patches = rearrange(patches, "b c h w -> (h w) b c")
        patches = patches + self.pos_embedding

        # Shuffle patches
        patches, forward_indexes, backward_indexes = self.shuffle(patches)

        # Add CLS token and pass through transformer
        patches = torch.cat([self.cls_token.expand(-1, patches.shape[1], -1), patches], dim=0)
        patches = rearrange(patches, "t b c -> b t c")

        # Apply transformer and normalization
        features = self.layer_norm(self.transformer(patches))
        features = rearrange(features, "b t c -> t b c")

        return features, backward_indexes


class MAEDecoder(nn.Module):
    def __init__(
        self,
        img_size: int = 32,
        patch_size: int = 2,
        emb_dim: int = 192,
        num_layers: int = 4,
        num_head: int = 3,
    ):
        """
        Masked Autoencoder (MAE) Decoder module.

        Args:
        - img_size (int): Size of the input image (assumed square).
        - patch_size (int): Size of the patches.
        - emb_dim (int): Embedding dimension.
        - num_layers (int): Number of transformer layers.
        - num_heads (int): Number of attention heads.
        """
        super(MAEDecoder, self).__init__()

        self.mask_token = torch.nn.Parameter(torch.zeros(1, 1, emb_dim))
        self.pos_embedding = torch.nn.Parameter(torch.zeros((img_size // patch_size) ** 2 + 1, 1, emb_dim))
        self.transformer = torch.nn.Sequential(*[Block(emb_dim, num_head) for _ in range(num_layers)])
        self.head = torch.nn.Linear(emb_dim, 3 * patch_size**2)
        self.patch2img = Rearrange(
            "(h w) b (c p1 p2) -> b c (h p1) (w p2)", p1=patch_size, p2=patch_size, h=img_size // patch_size
        )

        self._init_weights()

    def _init_weights(self):
        """Initialize weights for the mask token and position embedding."""
        trunc_normal_(self.mask_token, std=0.02)
        trunc_normal_(self.pos_embedding, std=0.02)

    def forward(self, features: torch.Tensor, backward_indexes: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the decoder.

        Args:
        - features (torch.Tensor): Encoded features from the encoder.
        - backward_indexes (torch.Tensor): Backward indexes from the shuffle operation.

        Returns:
        - Tuple[torch.Tensor, torch.Tensor]: Reconstructed image and the mask.
        """
        T = features.shape[0]

        # Adjust backward indexes and append mask tokens
        backward_indexes = torch.cat(
            [torch.zeros(1, backward_indexes.shape[1]).to(backward_indexes), backward_indexes + 1], dim=0
        )
        features = torch.cat(
            [features, self.mask_token.expand(backward_indexes.shape[0] - features.shape[0], features.shape[1], -1)],
            dim=0,
        )

        # Reorder features based on backward indexes and add position embeddings
        features = take_indexes(features, backward_indexes)
        features = features + self.pos_embedding

        # Pass through the transformer
        features = rearrange(features, "t b c -> b t c")
        features = self.transformer(features)
        features = rearrange(features, "b t c -> t b c")
        features = features[1:]  # remove global feature

        # Decode patches into image space
        patches = self.head(features)
        mask = torch.zeros_like(patches)
        mask[T - 1 :] = 1
        mask = take_indexes(mask, backward_indexes[1:] - 1)

        img = self.patch2img(patches)
        mask = self.patch2img(mask)

        return img, mask


class MaskedAutoEncoder(nn.Module):
    def __init__(
        self,
        img_size: int = 32,
        patch_size: int = 2,
        emb_dim: int = 192,
        encoder_layer: int = 12,
        encoder_head: int = 3,
        decoder_layer: int = 4,
        decoder_head: int = 3,
        mask_ratio: float = 0.75,
    ):
        """
        Masked Autoencoder (MAE) model combining encoder and decoder.

        Args:
        - img_size (int): Size of the input image.
        - patch_size (int): Size of the patches.
        - emb_dim (int): Embedding dimension.
        - encoder_layers (int): Number of transformer layers in the encoder.
        - encoder_heads (int): Number of attention heads in the encoder.
        - decoder_layers (int): Number of transformer layers in the decoder.
        - decoder_heads (int): Number of attention heads in the decoder.
        - mask_ratio (float): Ratio of patches to be masked.
        """
        super(MaskedAutoEncoder, self).__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.emb_dim = emb_dim
        self.encoder_layer = encoder_layer
        self.encoder_head = encoder_head
        self.decoder_layer = decoder_layer
        self.decoder_head = decoder_head
        self.mask_ratio = mask_ratio
        self.encoder = MAEEncoder(img_size, patch_size, emb_dim, encoder_layer, encoder_head, mask_ratio)
        self.decoder = MAEDecoder(img_size, patch_size, emb_dim, decoder_layer, decoder_head)

    def forward(self, image: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the MAE model.
        Args:
        - image (torch.Tensor): Input image tensor.
        Returns:
        - Tuple[torch.Tensor, torch.Tensor]: Reconstructed image and the mask.
        """
        features, backward_indexes = self.encoder(image)
        pred_img, mask = self.decoder(features, backward_indexes)
        return pred_img, mask


if __name__ == "__main__":
    # Testing the PatchShuffle and MaskedAutoEncoder
    patch_shuffle = PatchShuffle(0.75)
    sample_tensor = torch.rand(16, 2, 10)
    shuffled_patches, fwd_idx, bwd_idx = patch_shuffle(sample_tensor)
    print(shuffled_patches.shape)

    # Test the encoder and decoder
    input_img = torch.rand(2, 3, 32, 32)
    encoder = MAEEncoder()
    decoder = MAEDecoder()

    mae = MaskedAutoEncoder()
    # features, bwd_idx = encoder(input_img)
    # predicted_img, mask = decoder(features, bwd_idx)
    predicted_img, mask = mae(input_img)
    print(predicted_img.shape)
    print(torch.mean((predicted_img - input_img) ** 2 * mask / 0.75))
