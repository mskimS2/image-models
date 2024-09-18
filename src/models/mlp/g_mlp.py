import torch
from torch import nn
from torch.nn import functional as F


class SpatialGatingUnit(nn.Module):
    def __init__(self, hidden_dim: int, seq_len: int) -> None:
        super(SpatialGatingUnit, self).__init__()
        self.norm = nn.LayerNorm(hidden_dim)
        self.spatial_proj = nn.Conv1d(seq_len, seq_len, kernel_size=1)
        nn.init.constant_(self.spatial_proj.bias, 1.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u, v = x.chunk(2, dim=-1)
        v = self.norm(v)
        v = self.spatial_proj(v)
        return u * v


class gMLPLayer(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        ffn_dim: int,
        seq_len: int,
    ) -> None:
        super(gMLPLayer, self).__init__()
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.channel_proj1 = nn.Linear(hidden_dim, ffn_dim * 2)
        self.channel_proj2 = nn.Linear(ffn_dim, hidden_dim)
        self.sgu = SpatialGatingUnit(ffn_dim, seq_len)
        self.gelu = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        emb = self.layer_norm(x)
        emb = self.gelu(self.channel_proj1(emb))
        emb = self.sgu(emb)
        emb = self.channel_proj2(emb)
        return emb + x # resudual connection
    
    
class gMLPHaed(nn.Module):
    def __init__(
        self,
        hidden_dim: int = 256,
        ffn_dim: int = 512,
        seq_len: int = 256,
        num_layers: int = 6,
    ) -> None:
        super(gMLPHaed, self).__init__()
        self.layers = nn.Sequential(
            *[
                gMLPLayer(hidden_dim, ffn_dim, seq_len)
                for _ in range(num_layers)
            ],
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class gMLP(nn.Module):
    def __init__(
        self,
        img_size: int = 256,
        patch_size: int = 16,
        in_channels: int = 3,
        hidden_dim: int = 256,
        ffn_dim: int = 512,
        seq_len: int = 256,
        num_layers: int = 6,
        num_classes: int = 10,
    ) -> None:
        super(gMLP, self).__init__()
        assert img_size % patch_size == 0, "`img_size` must be divisible by the `patch_size`."
    
        self.patcher = nn.Conv2d(
            in_channels=in_channels, 
            out_channels=hidden_dim, 
            kernel_size=patch_size, 
            stride=patch_size,
        )
        self.gmlp = gMLPHaed(hidden_dim, ffn_dim, seq_len, num_layers)
        self.classifier = nn.Linear(hidden_dim, num_classes)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        patches = self.patcher(x)
        B, C, H, W = patches.shape
        patches = patches.permute(0, 2, 3, 1).view(B, H * W, C)
        emb = self.gmlp(patches).mean(dim=1)
        return self.classifier(emb)
    

if __name__ == '__main__':
    import torchsummary

    # test code
    gmlp = gMLP(
        img_size=256,
        patch_size=16,
        in_channels=3,
        hidden_dim=256,
        ffn_dim=512,
        seq_len=256,
        num_layers=6,
        num_classes=10,
    )

    torchsummary.summary(gmlp.cuda(), (3, 256, 256))
