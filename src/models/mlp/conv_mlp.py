import torch
from torch import nn
from typing import List
from mlp.layers import DropPath


class ConvTokenizer(nn.Module):
    def __init__(
        self, 
        in_channels: int = 3,
        embedding_dim: int = 64,
    ):
        super(ConvTokenizer, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(
                in_channels,
                embedding_dim // 2,
                kernel_size=(3, 3),
                stride=(2, 2),
                padding=(1, 1),
                bias=False,
            ),
            nn.BatchNorm2d(embedding_dim // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                embedding_dim // 2,
                embedding_dim // 2,
                kernel_size=(3, 3),
                stride=(1, 1),
                padding=(1, 1),
                bias=False,
            ),
            nn.BatchNorm2d(embedding_dim // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                embedding_dim // 2,
                embedding_dim,
                kernel_size=(3, 3),
                stride=(1, 1),
                padding=(1, 1),
                bias=False,
            ),
            nn.BatchNorm2d(embedding_dim),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(
                kernel_size=(3, 3),
                stride=(2, 2),
                padding=(1, 1),
                dilation=(1, 1),
            )
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class ConvMLPStageBlock(nn.Module):
    def __init__(
        self,
        num_layers: int = 2,
        embedding_dim_in: int = 64,
        embedding_dim_out: int = 128,
        hidden_dim: int = 128,
    ):
        super(ConvMLPStageBlock, self).__init__()
        self.conv_mlp_layers = nn.ModuleList(
            *[
                nn.Sequential(
                    nn.Conv2d(
                        embedding_dim_in,
                        hidden_dim,
                        kernel_size=(1, 1),
                        stride=(1, 1),
                        padding=(0, 0),
                        bias=False,
                    ),
                    nn.BatchNorm2d(hidden_dim),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(
                        hidden_dim, 
                        hidden_dim, 
                        kernel_size=(3, 3), 
                        stride=(1, 1), 
                        padding=(1, 1), 
                        bias=False,
                    ),
                    nn.BatchNorm2d(hidden_dim),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(
                        hidden_dim, 
                        embedding_dim_in,
                        kernel_size=(1, 1), 
                        stride=(1, 1), 
                        padding=(0, 0), 
                        bias=False,
                    ),
                    nn.BatchNorm2d(embedding_dim_in),
                    nn.ReLU(inplace=True)
                ) for _ in range(num_layers)
            ] 
        )
        self.downsample = nn.Conv2d(
            embedding_dim_in,
            embedding_dim_out,
            kernel_size=(3, 3),
            stride=(2, 2),
            padding=(1, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.conv_mlp_layers:
            x = x + layer(x)
        return self.downsample(x)


class ConvMLPStage(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        dim_feedforward: int = 2048,
        stochastic_depth_rate: float = 0.1,
    ):
        super(ConvMLPStage, self).__init__()
        self.layer_norm1 = nn.LayerNorm(embedding_dim)
        self.channel_mlp1 = MLP(
            embedding_dim_in=embedding_dim, 
            hidden_dim=dim_feedforward,
        )
        self.layer_norm2 = nn.LayerNorm(embedding_dim)
        self.connect = nn.Conv2d(
            embedding_dim,
            embedding_dim,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=(1, 1),
            groups=embedding_dim,
            bias=False,
        )
        self.connect_norm = nn.LayerNorm(embedding_dim)
        self.channel_mlp2 = MLP(
            embedding_dim_in=embedding_dim, 
            hidden_dim=dim_feedforward,
        )
        self.drop_path = DropPath(stochastic_depth_rate) if stochastic_depth_rate > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.drop_path(self.channel_mlp1(self.layer_norm1(x)))
        x = self.connect(self.connect_norm(x).permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        return x + self.drop_path(self.channel_mlp2(self.layer_norm2(x)))


class MLP(nn.Module):
    def __init__(
        self,
        embedding_dim_in: int,
        hidden_dim: int = None,
        embedding_dim_out: int = None,
    ):
        super(MLP, self).__init__()
        hidden_dim = hidden_dim or embedding_dim_in
        embedding_dim_out = embedding_dim_out or embedding_dim_in
        self.fc1 = nn.Linear(embedding_dim_in, hidden_dim)
        self.gelu = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, embedding_dim_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.gelu(self.fc1(x)))


class ConvDownsample(nn.Module):
    def __init__(
        self, 
        embedding_dim_in: int, 
        embedding_dim_out: int,
    ):
        super(ConvDownsample, self).__init__()
        self.downsample = nn.Conv2d(
            embedding_dim_in, 
            embedding_dim_out, 
            kernel_size=(3, 3),
            stride=(2, 2),
            padding=(1, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 3, 1, 2)
        x = self.downsample(x)
        return x.permute(0, 2, 3, 1)
    

class BasicStage(nn.Module):
    def __init__(
        self,
        num_layers: int,
        embedding_dims: int,
        mlp_ratio: float = 1.,
        stochastic_depth_rate: float = 0.1,
        downsample: bool = True,
    ):
        super(BasicStage, self).__init__()
        dpr = [x.item() for x in torch.linspace(0, stochastic_depth_rate, num_layers)]
        self.layers = nn.ModuleList(
            *[
                ConvMLPStage(
                    embedding_dim=embedding_dims[0],
                    dim_feedforward=int(embedding_dims[0] * mlp_ratio),
                    stochastic_depth_rate=dpr[i],
                ) for i in range(num_layers)
            ]
        )
        self.downsample_mlp = ConvDownsample(embedding_dims[0], embedding_dims[1]) if downsample else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return self.downsample_mlp(x)
    
    
class ConvMLP(nn.Module):
    def __init__(
        self,
        depth: List[int],
        d_model: List[int],
        expansion_factor: List[int],
        embedding_dim: int = 64,
        num_conv_block_layers: int = 3,
        classifier_head: bool = True,
        num_classes: int = 1000,
        *args, 
        **kwargs,
    ):
        super(ConvMLP, self).__init__()
        assert len(depth) == len(expansion_factor) == len(expansion_factor), \
            f"`depth`, `d_model` and `expansion_factor` must agree in size, {len(depth)}, {len(d_model)} and {len(expansion_factor)} passed."

        self.tokenizer = ConvTokenizer(embedding_dim=embedding_dim)
        self.conv_stages = ConvMLPStageBlock(
            num_conv_block_layers,
            embedding_dim_in=embedding_dim,
            hidden_dim=d_model[0],
            embedding_dim_out=d_model[0],
        )
        self.stages = nn.ModuleList(
            *[
                BasicStage(
                    num_blocks=depth[i],
                    embedding_dims=d_model[i:i + 2],
                    mlp_ratio=expansion_factor[i],
                    stochastic_depth_rate=0.1,
                    downsample=(i + 1 < len(depth))
                ) for i in range(len(depth))
            ]
        )
        self.head = None
        if classifier_head:
            self.norm = nn.LayerNorm(d_model[-1])
            self.head = nn.Linear(d_model[-1], num_classes)
            
        self.apply(self.init_weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.tokenizer(x)
        x = self.conv_stages(x)
        x = x.permute(0, 2, 3, 1)
        for stage in self.stages:
            x = stage(x)
            
        if self.head is None:
            return x
        
        B, _, _, C = x.shape
        x = x.reshape(B, -1, C)
        x = self.norm(x)
        x = x.mean(dim=1)
        return self.head(x)

    @staticmethod
    def init_weight(m):
        if isinstance(m, (nn.Linear, nn.Conv1d)):
            nn.init.trunc_normal_(m.weight, std=.02)
            if isinstance(m, (nn.Linear, nn.Conv1d)) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1.)
            nn.init.constant_(m.bias, 0.)