from . import gaussian_diffusion as gd
from .respace import SpacedDiffusion, space_timesteps
from .unet import UNetModel

NUM_CLASSES = 1000


def create_model_and_diffusion(cfg):
    return create_model(cfg), create_gaussian_diffusion(cfg)


def create_model(cfg):
    image_size = cfg["image_size"]
    if cfg["channel_mult"]:
        channel_mult = tuple(int(value) for value in cfg["channel_mult"].split(","))
    elif image_size == 128:
        channel_mult = (1, 1, 2, 3, 4)
    elif image_size == 64:
        channel_mult = (1, 2, 3, 4)
    else:
        raise ValueError(f"unsupported image size: {image_size}")

    attention_ds = tuple(
        image_size // int(resolution)
        for resolution in cfg["attention_resolutions"].split(",")
    )
    return UNetModel(
        image_size=image_size,
        in_channels=cfg["in_channels"],
        model_channels=cfg["num_channels"],
        out_channels=cfg["in_channels"] * (2 if cfg["learn_sigma"] else 1),
        num_res_blocks=cfg["num_res_blocks"],
        attention_resolutions=attention_ds,
        dropout=cfg["dropout"],
        channel_mult=channel_mult,
        num_classes=NUM_CLASSES if cfg["class_cond"] else None,
        use_checkpoint=cfg["use_checkpoint"],
        num_heads=cfg["num_heads"],
        num_head_channels=cfg["num_head_channels"],
        num_heads_upsample=cfg["num_heads_upsample"],
        use_scale_shift_norm=cfg["use_scale_shift_norm"],
        resblock_updown=cfg["resblock_updown"],
        use_new_attention_order=cfg["use_new_attention_order"],
    )


def create_gaussian_diffusion(cfg):
    steps = cfg["diffusion_steps"]
    betas = gd.get_named_beta_schedule(cfg["noise_schedule"], steps)
    if cfg["use_kl"]:
        loss_type = gd.LossType.RESCALED_KL
    elif cfg["rescale_learned_sigmas"]:
        loss_type = gd.LossType.RESCALED_MSE
    else:
        loss_type = gd.LossType.MSE

    timestep_respacing = cfg["timestep_respacing"] or [steps]

    return SpacedDiffusion(
        use_timesteps=space_timesteps(steps, timestep_respacing),
        betas=betas,
        model_mean_type=(
            gd.ModelMeanType.START_X
            if cfg["predict_xstart"]
            else gd.ModelMeanType.EPSILON
        ),
        model_var_type=(
            gd.ModelVarType.LEARNED_RANGE
            if cfg["learn_sigma"]
            else gd.ModelVarType.FIXED_LARGE
        ),
        loss_type=loss_type,
        rescale_timesteps=cfg["rescale_timesteps"],
    )
