import torch
from diffusers import UNet2DConditionModel

def build_ddpm_model(size=128):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = UNet2DConditionModel(
        sample_size=size,
        in_channels=1,
        out_channels=1,
        block_out_channels=(64, 128, 128),
        down_block_types=("DownBlock2D", "AttnDownBlock2D", "DownBlock2D"),
        up_block_types=("UpBlock2D", "AttnUpBlock2D", "UpBlock2D"),
        cross_attention_dim=128,
        num_class_embeds=None,
    ).to(device)
    return model
