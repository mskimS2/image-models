import torch
import numpy as np

from mlp.mlp_mixer import MlpMixer
from mlp.res_mlp import ResMLP
from mlp.conv_mixer import ConvMixer
from transformer.vit import VisionTransformer
from utils import get_parameters


def get_model(config):
    if config.model_name == "mlpmixer":
        return MlpMixer(
            in_channels=config.in_channels,
            img_size=config.img_size,
            patch_size=config.patch_size,
            tokens_mlp_dim=config.tokens_mlp_dim,
            channels_mlp_dim=config.channels_mlp_dim,
            hidden_dim=config.hidden_dim,
            dropout_p=config.dropout_p,
            n_layers=config.num_layers,
            n_classes=config.num_classes,
        ).to(config.device)
    elif config.model_name == "resmlp":
        return ResMLP(
            img_size=config.img_size,
            in_channels=config.in_channels,
            patch_size=config.patch_size,
            input_dim=config.input_dim,
            hidden_dim=config.hidden_dim,
            dropout_p=config.dropout_p,
            n_layers=config.num_layers,
            num_classes=config.num_classes,
        ).to(config.device)
    elif config.model_name == "convmixer":
        return ConvMixer(
            in_channels=config.in_channels,
            hidden_dim=config.hidden_dim,
            n_layers=config.num_layers,
            kernel_size=config.kennel_size,
            patch_size=config.patch_size,
            num_classes=config.num_classes,
        ).to(config.device)
    elif config.model_name == "vit":
        return VisionTransformer(
            img_size=config.img_size,
            in_channels=config.in_channels,
            patch_size=config.patch_size,
            embed_dim=config.embed_dim,
            n_layers=config.num_layers,
            n_heads=config.num_nheads,
            dropout_p=config.dropout_p,
            mlp_ratio=config.mlp_ratio,
            num_classes=config.num_classes,
        ).to(config.device)
    
    raise ValueError(f"No models found with {config.model_name}.")


def get_parameters(model: torch.nn.Module):
    parameters = filter(lambda p: p.requires_grad, model.parameters())
    parameters = sum([np.prod(p.size()) for p in parameters]) / 1_000
    print('Trainable Parameters: %.3fK' % parameters)
    return parameters