import torch
from torch import nn
from einops import repeat, rearrange
from einops.layers.torch import Rearrange


class MAELinearProbing(nn.Module):
    def __init__(self, backbone: nn.Module, num_classes: int = 10):
        super(MAELinearProbing, self).__init__()
        self.num_classes = num_classes
        self.backbone = backbone
        self.cls_token = backbone.cls_token
        self.pos_embedding = backbone.pos_embedding
        self.patchify = backbone.patchify
        self.transformer = backbone.transformer
        self.layer_norm = backbone.layer_norm
        self.head = torch.nn.Linear(self.pos_embedding.shape[-1], num_classes)

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        patches = self.patchify(img)
        patches = rearrange(patches, "b c h w -> (h w) b c")
        patches = patches + self.pos_embedding
        patches = torch.cat([self.cls_token.expand(-1, patches.shape[1], -1), patches], dim=0)
        patches = rearrange(patches, "t b c -> b t c")
        features = self.layer_norm(self.transformer(patches))
        features = rearrange(features, "b t c -> t b c")
        return self.head(features[0])


if __name__ == "__main__":
    from transformer.mae import MaskedAutoEncoder

    mae = MaskedAutoEncoder(
        image_size=32,
        patch_size=2,
        emb_dim=192,
        encoder_layer=12,
        encoder_head=3,
        decoder_layer=4,
        decoder_head=3,
        mask_ratio=0.75,
    )
    mae_linear_probing = MAELinearProbing(mae.encoder, num_classes=10)
    x = torch.randn(1, 3, 32, 32)
    print(mae_linear_probing(x).shape)
