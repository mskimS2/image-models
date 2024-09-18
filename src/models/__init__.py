from models.cnn.cnn import CNN
from models.mlp.mlp_mixer import MlpMixer
from models.mlp.res_mlp import ResMLP
from models.mlp.conv_mixer import ConvMixer
from models.mlp.g_mlp import gMLP
from models.mlp.gfnet import GFNet

from models.transformer.vit import VisionTransformer

from configs.config import Config


def get_model(config: Config):
    if config.model_name == "cnn":
        return CNN(num_classes=config.num_classes)

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
        )

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
        )

    elif config.model_name == "convmixer":
        return ConvMixer(
            in_channels=config.in_channels,
            hidden_dim=config.hidden_dim,
            n_layers=config.num_layers,
            kernel_size=config.kennel_size,
            patch_size=config.patch_size,
            num_classes=config.num_classes,
        )

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
        )

    elif config.model_name == "gmlp":
        return gMLP(
            img_size=config.img_size,
            patch_size=config.patch_size,
            in_channels=config.in_channels,
            hidden_dim=config.hidden_dim,
            ffn_dim=config.ffn_dim,
            seq_len=config.seq_len,
            num_layers=config.num_layers,
            num_classes=config.num_classes,
        )

    elif config.model_name == "gfnet":
        return GFNet(
            img_size=config.img_size,
            patch_size=config.patch_size,
            in_channels=config.in_channels,
            num_classes=config.num_classes,
        )

    raise ValueError(f"No models found with {config.model_name}.")
