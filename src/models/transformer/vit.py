import torch
from torch import nn


class VisionTransformer(nn.Module):
    def __init__(
        self,
        img_size: int = 384,
        patch_size: int = 16,
        in_channels: int = 3,
        embed_dim: int = 768,
        n_heads: int = 12,
        mlp_ratio: int = 4.,
        dropout_p: float = 0.,
        n_layers: int = 12,
        num_classes: int = 1000,
    ):
        super(VisionTransformer, self).__init__()

        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=embed_dim,
        )
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(
            torch.zeros(1, 1 + self.patch_embed.n_patches, embed_dim)
        )
        self.pos_drop = nn.Dropout(dropout_p)
        self.encoder_layers = nn.Sequential(
            *[
                EncoderBlock(
                    embed_dim=embed_dim,
                    n_heads=n_heads,
                    mlp_ratio=mlp_ratio,
                    dropout_p=dropout_p
                ) for _ in range(n_layers)
            ]
        )
        self.norm = nn.LayerNorm(embed_dim, eps=1e-6)
        self.classifier = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        n_samples = x.shape[0]
        x = self.patch_embed(x)  # (n_samples, n_patches, embed_dim)
        cls_token = self.cls_token.expand(
            n_samples, -1, -1)  # (n_samples, 1, embed_dim)

        # (n_samples, 1+n_patches, embed_dim)
        x = torch.cat((cls_token, x), dim=1)
        x = x + self.pos_embed  # (n_samples, 1+n_patches, embed_dim)
        x = self.pos_drop(x)

        x = self.encoder_layers(x)  # (n_samples, 577, 768)
        x = self.norm(x)

        cls_token_final = x[:, 0]  # just tje CLS token
        x = self.classifier(cls_token_final)

        return x


class PatchEmbed(nn.Module):

    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        in_channels: int = 3,
        embed_dim: int = 768
    ):
        super(PatchEmbed, self).__init__()

        self.img_size = img_size
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(
            in_channels=in_channels,
            out_channels=embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )

    def forward(self, x):
        # (batch_size, embed_dim, n_patches ** 0.5, n_patches ** 0.5)
        x = self.proj(x)
        x = x.flatten(2)        # (batch_size, embed_dim, n_patches)
        x = x.transpose(1, 2)   # (batch_size, n_patches, embed_dim)

        return x


class Attention(nn.Module):
    def __init__(
        self,
        dim: int,
        n_heads: int = 12,
        dropout_p: float = 0.
    ):
        super(Attention, self).__init__()
        self.n_heads = n_heads
        self.dim = dim
        self.head_dim = dim // n_heads
        self.scale = self.head_dim ** -0.5  # 1 / root(self.head_dim)
        self.qkv = nn.Linear(dim, dim*3, bias=True)
        self.attn_drop = nn.Dropout(dropout_p)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(dropout_p)

    def forward(self, x):
        n_samples, n_tokens, dim = x.shape
        if dim != self.dim:
            raise ValueError

        qkv = self.qkv(x)  # (n_samples, n_patches+1, dim*3)
        qkv = qkv.reshape(
            n_samples, n_tokens, 3, self.n_heads, self.head_dim
        )  # (n_samples, n_patches+1, 3, n_heads, head_dim)

        # (3, n_samples, n_heads, n_patches+1, head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)

        # (n_samples, n_heads, head_dim, n_patches+1)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # (n_samples, n_heads, n_patches+1, n_patches+1)
        k_t = k.transpose(-2, -1)

        # attention (n_samples, n_heads, n_patches+1, n_patches+1)
        dot = (q @ k_t) * self.scale
        attn = dot.softmax(dim=-1)
        attn = self.attn_drop(attn)

        # (n_samples, n_heads, n_patches+1, head_dim)
        weighted_avg = attn @ v

        # (n_samples, n_patches+1, n_heads, head_dim)
        weighted_avg = weighted_avg.transpose(1, 2)

        # concat (n_samples, n_patches+1, dim)
        weighted_avg = weighted_avg.flatten(2)

        # linear projection (n_samples, n_patches+1, dim)
        x = self.proj(weighted_avg)
        x = self.proj_drop(x)

        return x


class MLP(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        dropout_p: float = 0.
    ):
        super(MLP, self).__init__()

        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_dim),
            nn.Dropout(dropout_p)
        )

    def forward(self, x):
        return self.mlp(x)


class EncoderBlock(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        n_heads: int,
        mlp_ratio: int = 4.0,
        dropout_p: float = 0.,
    ):
        super(EncoderBlock, self).__init__()
        self.pre_norm1 = nn.LayerNorm(embed_dim, eps=1e-6)
        self.attn = Attention(
            embed_dim,
            n_heads=n_heads,
            dropout_p=dropout_p
        )
        self.pre_norm2 = nn.LayerNorm(embed_dim, eps=1e-6)
        hidden_features = int(embed_dim * mlp_ratio)  # 3072(MLP size)
        self.mlp = MLP(
            in_dim=embed_dim,
            hidden_dim=hidden_features,
            out_dim=embed_dim,
        )

    def forward(self, x):
        # residual
        x = x + self.attn(self.pre_norm1(x))
        x = x + self.mlp(self.pre_norm2(x))

        return x


if __name__ == '__main__':
    from torchsummary import summary

    cifar10_config = {
        "img_size": 32,
        "in_channels": 3,
        "patch_size": 4,
        "embed_dim": 512,
        "n_layers": 6,
        "n_heads": 8,
        "dropout_p": 0.1,
        "mlp_ratio": 4,
        "num_classes": 10,
    }

    vit = VisionTransformer(**cifar10_config)
    summary(vit.cuda(), (3, 32, 32))
