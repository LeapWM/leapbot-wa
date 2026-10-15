import torch
from omegaconf import DictConfig, OmegaConf


def create_leapbot_wa(
    model_id: str,
    tokenizer_model_id: str,
    video_dit_config,
    tokenizer_max_len: int = 512,
    load_text_encoder: bool = True,
    proprio_dim: int | None = None,
    action_dit_config=None,
    action_dit_pretrained_path: str | None = None,
    skip_dit_load_from_pretrain: bool = False,
    video_scheduler=None,
    action_scheduler=None,
    jepa_scheduler=None,
    loss=None,
    jepa_supervision=None,
    jepa_expert=None,
    jepa_adapter=None,
    mot_checkpoint_mixed_attn: bool = True,
    redirect_common_files: bool = True,
    model_dtype: torch.dtype = torch.bfloat16,
    device: str = "cuda",
):
    from .models.wan22.leapbot_wa import LeapBotWA

    if isinstance(video_dit_config, DictConfig):
        video_dit_config = OmegaConf.to_container(video_dit_config, resolve=True)
    if not isinstance(video_dit_config, dict):
        raise ValueError(f"`video_dit_config` must resolve to a dict, got {type(video_dit_config)}")

    if isinstance(action_dit_config, DictConfig):
        action_dit_config = OmegaConf.to_container(action_dit_config, resolve=True)
    if action_dit_config is None:
        action_dit_config = {}
    if not isinstance(action_dit_config, dict):
        raise ValueError(f"`action_dit_config` must resolve to a dict, got {type(action_dit_config)}")

    if isinstance(video_scheduler, DictConfig):
        video_scheduler = OmegaConf.to_container(video_scheduler, resolve=True)
    if video_scheduler is None:
        video_scheduler = {}
    if not isinstance(video_scheduler, dict):
        raise ValueError(f"`video_scheduler` must be dict-like, got {type(video_scheduler)}")

    if isinstance(action_scheduler, DictConfig):
        action_scheduler = OmegaConf.to_container(action_scheduler, resolve=True)
    if action_scheduler is None:
        raise ValueError("`action_scheduler` is required for LeapBotWA.")
    if not isinstance(action_scheduler, dict):
        raise ValueError(f"`action_scheduler` must be dict-like, got {type(action_scheduler)}")
    required_action_scheduler_keys = {"train_shift", "infer_shift", "num_train_timesteps"}
    missing_keys = required_action_scheduler_keys - set(action_scheduler.keys())
    if missing_keys:
        raise ValueError(
            f"`action_scheduler` missing required keys: {sorted(missing_keys)}. "
            "Expected keys: train_shift, infer_shift, num_train_timesteps."
        )

    if isinstance(jepa_scheduler, DictConfig):
        jepa_scheduler = OmegaConf.to_container(jepa_scheduler, resolve=True)
    if jepa_scheduler is None:
        jepa_scheduler = {}
    if not isinstance(jepa_scheduler, dict):
        raise ValueError(f"`jepa_scheduler` must be dict-like, got {type(jepa_scheduler)}")

    if isinstance(loss, DictConfig):
        loss = OmegaConf.to_container(loss, resolve=True)
    if loss is None:
        loss = {}
    if not isinstance(loss, dict):
        raise ValueError(f"`loss` must be dict-like, got {type(loss)}")

    branches = {"video": False, "jepa": True, "action": True}

    if isinstance(jepa_supervision, DictConfig):
        jepa_supervision = OmegaConf.to_container(jepa_supervision, resolve=True)
    if jepa_supervision is None:
        jepa_supervision = {}
    if not isinstance(jepa_supervision, dict):
        raise ValueError(f"`jepa_supervision` must be dict-like, got {type(jepa_supervision)}")

    if isinstance(jepa_expert, DictConfig):
        jepa_expert = OmegaConf.to_container(jepa_expert, resolve=True)
    if jepa_expert is None:
        jepa_expert = {}
    if not isinstance(jepa_expert, dict):
        raise ValueError(f"`jepa_expert` must be dict-like, got {type(jepa_expert)}")

    if isinstance(jepa_adapter, DictConfig):
        jepa_adapter = OmegaConf.to_container(jepa_adapter, resolve=True)
    if jepa_adapter is None:
        jepa_adapter = {}
    if not isinstance(jepa_adapter, dict):
        raise ValueError(f"`jepa_adapter` must be dict-like, got {type(jepa_adapter)}")

    # Architecture constants of the released mixed-encoder/mixed-adapter model.
    for config, fixed in (
        (jepa_supervision, {"enabled": True, "mode": "jepa_expert", "online_encoder": True}),
        (jepa_adapter, {"enabled": True, "freeze": True}),
        (jepa_expert, {"use_t_mod": True, "position_mode": "vjepa3d_rope",
                       "attention_mode": "action_reads_video_jepa"}),
    ):
        for key, value in fixed.items():
            if key in config:
                raise ValueError(f"{key} is fixed by the released architecture, not a configuration option")
            config[key] = value


    return LeapBotWA.from_wan22_pretrained(
        device=device,
        torch_dtype=model_dtype,
        model_id=model_id,
        tokenizer_model_id=tokenizer_model_id,
        tokenizer_max_len=int(tokenizer_max_len),
        load_text_encoder=bool(load_text_encoder),
        proprio_dim=(None if proprio_dim is None else int(proprio_dim)),
        redirect_common_files=bool(redirect_common_files),
        video_dit_config=video_dit_config,
        action_dit_config=action_dit_config,
        action_dit_pretrained_path=action_dit_pretrained_path,
        skip_dit_load_from_pretrain=bool(skip_dit_load_from_pretrain),
        mot_checkpoint_mixed_attn=bool(mot_checkpoint_mixed_attn),
        video_train_shift=float(video_scheduler.get("train_shift", 5.0)),
        video_infer_shift=float(video_scheduler.get("infer_shift", 5.0)),
        video_num_train_timesteps=int(video_scheduler.get("num_train_timesteps", 1000)),
        action_train_shift=float(action_scheduler["train_shift"]),
        action_infer_shift=float(action_scheduler["infer_shift"]),
        action_num_train_timesteps=int(action_scheduler["num_train_timesteps"]),
        jepa_train_shift=float(jepa_scheduler.get("train_shift", video_scheduler.get("train_shift", 5.0))),
        jepa_infer_shift=float(
            jepa_scheduler.get(
                "infer_shift",
                jepa_scheduler.get("train_shift", video_scheduler.get("infer_shift", 5.0)),
            )
        ),
        jepa_num_train_timesteps=int(jepa_scheduler.get("num_train_timesteps", video_scheduler.get("num_train_timesteps", 1000))),
        loss_lambda_video=float(loss.get("lambda_video", 1.0)),
        loss_lambda_action=float(loss.get("lambda_action", 1.0)),
        loss_lambda_jepa=(
            float(loss["lambda_jepa"])
            if loss.get("lambda_jepa", None) is not None
            else None
        ),
        loss_lambda_jepa_feature=float(loss.get("lambda_jepa_feature", 0.0)),
        branches=branches,
        jepa_supervision=jepa_supervision,
        jepa_expert=jepa_expert,
        jepa_adapter=jepa_adapter,
    )
