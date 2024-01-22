import math
import torch
import torch.nn.functional as F
from torch import nn
from functools import partial
from collections import OrderedDict
from typing import List, Dict
from mlp.layer import DropPath


class MLP(nn.Module):
    def __init__(
        self, 
        in_features: int, 
        hidden_features: int = None, 
        out_features: int = None, 
        dropout_p: float = 0.,
    ):
        super(MLP, self).__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.in_features = in_features
        self.hidden_features = hidden_features
        self.out_features = out_features
        self.dropout_p = dropout_p
        
        self.layers = nn.Sequential(
            nn.Linear(in_features, hidden_features),
            nn.GELU(),
            nn.Dropout(dropout_p),
            nn.Linear(hidden_features, out_features),
            nn.Dropout(dropout_p),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class GlobalFilter(nn.Module):
    def __init__(
        self, 
        dim: int, 
        h: int = 14, 
        w: int = 8,
    ):
        super(GlobalFilter, self).__init__()
        self.dim = dim
        self.w = w
        self.h = h
        self.complex_weight = nn.Parameter(torch.randn(h, w, dim, 2, dtype=torch.float32) * 0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, H, W, C = x.shape
        x = torch.fft.rfft2(x, dim=(1, 2), norm='ortho')
        weight = torch.view_as_complex(self.complex_weight)
        x = x * weight
        return torch.fft.irfft2(x, s=(H, W), dim=(1, 2), norm='ortho')


class Block(nn.Module):
    def __init__(
        self, 
        dim: int, 
        mlp_ratio: float = 4., 
        dropout_p: float = 0., 
        drop_path: float = 0.,  
        h: int = 14,  
        w: int = 8,
    ) -> None:
        super(Block, self).__init__()
        self.dim = dim
        self.mlp_ratio = mlp_ratio
        self.dropout_p = dropout_p
        self.drop_path = drop_path
        self.h = h
        self.w = w

        self.layers = nn.Sequential(
            nn.LayerNorm(dim),
            GlobalFilter(dim, h=h, w=w),
            nn.LayerNorm(dim),
            MLP(in_features=dim, hidden_features=int(dim * mlp_ratio), dropout_p=dropout_p),
            DropPath(drop_path) if drop_path > 0. else nn.Identity(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.layers(x)
        

class BlockLayerScale(nn.Module):
    def __init__(
        self, 
        dim: int, 
        mlp_ratio: float = 4., 
        dropout_p: float = 0., 
        drop_path: float = 0., 
        h: int = 14, 
        w: int = 8, 
        init_values: float = 1e-5,
    ) -> None:
        super(BlockLayerScale, self).__init__()
        self.dim = dim
        self.mlp_ratio = mlp_ratio
        self.dropout_p = dropout_p
        self.drop_path = drop_path
        self.h = h
        self.w = w
        self.init_values = init_values
        
        self.layers = nn.Sequential(
            nn.LayerNorm(dim),
            GlobalFilter(dim, h=h, w=w),
            nn.LayerNorm(dim),
            (
                nn.Parameter(init_values * torch.ones((dim)),requires_grad=True) * 
                MLP(in_features=dim, hidden_features=int(dim * mlp_ratio), dropout_p=dropout_p)
            ),
            DropPath(drop_path) if drop_path > 0. else nn.Identity(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.layers(x)
    

class PatchEmbed(nn.Module):
    def __init__(
        self, 
        img_size: int = 224, 
        patch_size: int = 16, 
        in_chans: int = 3, 
        embed_dim: int = 768,
    ) -> None:
        super(PatchEmbed, self).__init__()
        self.img_size = (img_size, img_size)
        self.patch_size = (patch_size, patch_size)
        self.num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        # FIXME look at relaxing size constraints
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        return self.proj(x).flatten(2).transpose(1, 2)


class DownLayer(nn.Module):
    def __init__(
        self, 
        img_size: int = 56, 
        dim_in: int = 64, 
        dim_out: int = 128
    ) -> None:
        super(DownLayer, self).__init__()
        self.img_size = img_size
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.proj = nn.Conv2d(dim_in, dim_out, kernel_size=2, stride=2)
        self.num_patches = img_size * img_size // 4

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.size()
        x = x.view(B, self.img_size, self.img_size, C).permute(0, 3, 1, 2)
        x = self.proj(x).permute(0, 2, 3, 1)
        return x.reshape(B, -1, self.dim_out)


class GFNet(nn.Module):
    def __init__(
        self, 
        img_size: int = 224, 
        patch_size: int = 16, 
        in_chans: int = 3, 
        num_classes: int = 1000, 
        embed_dim: int = 768, 
        depth: int = 12,
        mlp_ratio: float = 4., 
        representation_size: int = None, 
        uniform_drop: bool = False,
        dropout_p: float = 0., 
        drop_path_rate: float = 0., 
        dropcls: float=0,
    ) -> None:
        """
        Args:
            img_size (int, tuple): input image size
            patch_size (int, tuple): patch size
            in_chans (int): number of input channels
            num_classes (int): number of classes for classification head
            embed_dim (int): embedding dimension
            depth (int): depth of transformer
            num_heads (int): number of attention heads
            mlp_ratio (int): ratio of mlp hidden dim to embedding dim
            qkv_bias (bool): enable bias for qkv if True
            qk_scale (float): override default qk scale of head_dim ** -0.5 if set
            representation_size (Optional[int]): enable and set representation layer (pre-logits) to this value if set
            dropout_p (float): dropout rate
            attn_drop_rate (float): attention dropout rate
            drop_path_rate (float): stochastic depth rate
            hybrid_backbone (nn.Module): CNN backbone to use in-place of PatchEmbed module
            norm_layer: (nn.Module): normalization layer
        """
        super().__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.patch_embed = PatchEmbed(
                img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        self.pos_drop = nn.Dropout(p=dropout_p)

        h = img_size // patch_size
        w = h // 2 + 1

        if uniform_drop:
            print('using uniform droppath with expect rate', drop_path_rate)
            dpr = [drop_path_rate for _ in range(depth)]  # stochastic depth decay rule
        else:
            print('using linear droppath with expect rate', drop_path_rate * 0.5)
            dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        # dpr = [drop_path_rate for _ in range(depth)]  # stochastic depth decay rule
        
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, 
                mlp_ratio=mlp_ratio,
                dropout_p=dropout_p, 
                drop_path=dpr[i], 
                h=h, 
                w=w,
            ) for i in range(depth)
        ])
        
        self.norm = nn.LayerNorm(embed_dim)

        # Representation layer
        if representation_size > 0:
            self.num_features = representation_size
            self.pre_logits = nn.Sequential(OrderedDict([
                ('fc', nn.Linear(embed_dim, representation_size)),
                ('act', nn.Tanh())
            ]))
        else:
            self.pre_logits = nn.Identity()

        # Classifier head
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()
        if dropcls > 0:
            print('dropout %.2f before classifier' % dropcls)
            self.final_dropout = nn.Dropout(p=dropcls)
        else:
            self.final_dropout = nn.Identity()

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self) -> Dict[str]:
        return {'pos_embed', 'cls_token'}

    def get_classifier(self) -> nn.Module:
        return self.head

    def reset_classifier(self, num_classes: int) -> None:
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch_embed(x)
        x = x + self.pos_embed
        x = self.pos_drop(x)
        for blk in self.blocks:
            x = blk(x)
        return self.norm(x).mean(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.forward_features(x)
        x = self.final_dropout(x)
        return self.head(x)


class GFNetPyramid(nn.Module):
    def __init__(
        self, 
        img_size=224, 
        patch_size=4, 
        num_classes=1000, 
        embed_dim=[64, 128, 256, 512], 
        depth=[2,2,10,4],
        mlp_ratio=[4., 4., 4., 4.],
        dropout_p=0., 
        drop_path_rate=0., 
        init_values=0.001, 
        no_layerscale=False, 
        dropcls=0,
    ) -> None:
        """
        Args:
            img_size (int, tuple): input image size
            patch_size (int, tuple): patch size
            in_chans (int): number of input channels
            num_classes (int): number of classes for classification head
            embed_dim (int): embedding dimension
            depth (int): depth of transformer
            num_heads (int): number of attention heads
            mlp_ratio (int): ratio of mlp hidden dim to embedding dim
            qkv_bias (bool): enable bias for qkv if True
            qk_scale (float): override default qk scale of head_dim ** -0.5 if set
            representation_size (Optional[int]): enable and set representation layer (pre-logits) to this value if set
            dropout_p (float): dropout rate
            attn_drop_rate (float): attention dropout rate
            drop_path_rate (float): stochastic depth rate
            hybrid_backbone (nn.Module): CNN backbone to use in-place of PatchEmbed module
            norm_layer: (nn.Module): normalization layer
        """
        super().__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim[-1]  # num_features for consistency with other models
        patch_embed = PatchEmbed(
            img_size=img_size, 
            patch_size=patch_size, 
            in_chans=3, 
            embed_dim=embed_dim[0],
        )
        num_patches = patch_embed.num_patches
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim[0]))
        self.patch_embed = nn.ModuleList(patch_embed)

        sizes = [56, 28, 14, 7]
        for i in range(4):
            sizes[i] = sizes[i] * img_size // 224

        for i in range(3):
            patch_embed = DownLayer(sizes[i], embed_dim[i], embed_dim[i+1])
            num_patches = patch_embed.num_patches
            self.patch_embed.append(patch_embed)

        self.pos_drop = nn.Dropout(p=dropout_p)
        self.blocks = nn.ModuleList()

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depth))]  # stochastic depth decay rule
        cur = 0
        for i in range(4):
            h = sizes[i]
            w = h // 2 + 1

            if no_layerscale:
                print("using standard block")
                blk = nn.Sequential(
                    *[
                        Block(
                            dim=embed_dim[i], 
                            mlp_ratio=mlp_ratio[i],
                            dropout_p=dropout_p, 
                            drop_path=dpr[cur + j], 
                            h=h, 
                            w=w,
                        ) for j in range(depth[i])
                    ],
                )
            else:
                print("using layerscale block")
                blk = nn.Sequential(
                    *[
                        BlockLayerScale(
                            dim=embed_dim[i], 
                            mlp_ratio=mlp_ratio[i],
                            drop=dropout_p, 
                            drop_path=dpr[cur + j], 
                            h=h, 
                            w=w, 
                            init_values=init_values,
                        ) for j in range(depth[i])
                    ],
                )
            self.blocks.append(blk)
            cur += depth[i]

        # Classifier head
        self.norm = nn.LayerNorm(embed_dim[-1])
        self.head = nn.Linear(self.num_features, num_classes)
        if dropcls > 0:
            print('dropout %.2f before classifier' % dropcls)
            self.final_dropout = nn.Dropout(p=dropcls)
        else:
            self.final_dropout = nn.Identity()

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self) -> Dict[str]:
        return {'pos_embed', 'cls_token'}

    def get_classifier(self) -> nn.Module:
        return self.head

    def reset_classifier(self, num_classes: int) -> None:
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        for i in range(4):
            x = self.patch_embed[i](x)
            if i == 0:
                x = x + self.pos_embed
            x = self.blocks[i](x)

        return self.norm(x).mean(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.forward_features(x)
        x = self.final_dropout(x)
        return self.head(x)


def resize_pos_embed(posemb, posemb_new):
    # Rescale the grid of position embeddings when loading from state_dict. Adapted from
    # https://github.com/google-research/vision_transformer/blob/00883dd691c63a6830751563748663526e811cee/vit_jax/checkpoint.py#L224
    ntok_new = posemb_new.shape[1]
    if True:
        posemb_tok, posemb_grid = posemb[:, :1], posemb[0, 1:]
        ntok_new -= 1
    else:
        posemb_tok, posemb_grid = posemb[:, :0], posemb[0]
    gs_old = int(math.sqrt(len(posemb_grid)))
    gs_new = int(math.sqrt(ntok_new))
    posemb_grid = posemb_grid.reshape(1, gs_old, gs_old, -1).permute(0, 3, 1, 2)
    posemb_grid = F.interpolate(posemb_grid, size=(gs_new, gs_new), mode='bilinear')
    posemb_grid = posemb_grid.permute(0, 2, 3, 1).reshape(1, gs_new * gs_new, -1)
    posemb = torch.cat([posemb_tok, posemb_grid], dim=1)
    return posemb


def checkpoint_filter_fn(state_dict, model):
    """ convert patch embedding weight from manual patchify + linear proj to conv"""
    out_dict = {}
    if 'model' in state_dict:
        # For deit models
        state_dict = state_dict['model']
    for k, v in state_dict.items():
        if 'patch_embed.proj.weight' in k and len(v.shape) < 4:
            # For old models that I trained prior to conv based patchification
            O, I, H, W = model.patch_embed.proj.weight.shape
            v = v.reshape(O, -1, H, W)
        elif k == 'pos_embed' and v.shape != model.pos_embed.shape:
            # To resize pos embedding when using model at different size from pretrained weights
            v = resize_pos_embed(v, model.pos_embed)
        out_dict[k] = v
    return out_dict