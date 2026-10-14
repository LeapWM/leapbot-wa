import inspect
from typing import Any, Optional, Sequence, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

from leapbot_wa.utils.logging_config import get_logger

from .action_dit import ActionDiT
from .helpers.loader import load_wan22_ti2v_5b_components
from .jepa_adapter import FrozenJepaAdapter
from .jepa_state_expert import JEPAMoTStateExpert
from .mot import MoT
from .schedulers.scheduler_continuous import WanContinuousFlowMatchScheduler
from .vjepa_online_encoder import DEFAULT_VJEPA2_ROOT, DEFAULT_VJEPA2_CHECKPOINT, PerViewVJEPAEncoder, VJEPAOnlineEncoderWrapper

logger = get_logger(__name__)




class LeapBotWA(torch.nn.Module):
    """Released mixed V-JEPA/adapter action model."""

    def __init__(
        self,
        video_expert: Optional[nn.Module],
        action_expert: ActionDiT,
        mot: MoT,
        vae,
        text_encoder=None,
        tokenizer=None,
        text_dim: Optional[int] = None,
        proprio_dim: Optional[int] = None,
        device: str = "cpu",
        torch_dtype: torch.dtype = torch.float32,
        video_train_shift: float = 5.0,
        video_infer_shift: float = 5.0,
        video_num_train_timesteps: int = 1000,
        action_train_shift: float = 5.0,
        action_infer_shift: float = 5.0,
        action_num_train_timesteps: int = 1000,
        jepa_train_shift: float = 5.0,
        jepa_infer_shift: float = 5.0,
        jepa_num_train_timesteps: int = 1000,
        loss_lambda_video: float = 1.0,
        loss_lambda_action: float = 1.0,
        loss_lambda_jepa: float = 0.0,
        loss_lambda_jepa_feature: float = 0.0,
        vjepa_online_encoder: Optional[VJEPAOnlineEncoderWrapper] = None,
        per_view_vjepa_encoder: Optional[PerViewVJEPAEncoder] = None,
        jepa_state_expert: Optional[JEPAMoTStateExpert] = None,
        jepa_adapter: Optional[FrozenJepaAdapter] = None,
        jepa_supervision_config: Optional[dict[str, Any]] = None,
        jepa_expert_config: Optional[dict[str, Any]] = None,
        jepa_adapter_config: Optional[dict[str, Any]] = None,
        branch_config: Optional[dict[str, Any]] = None,
    ):
        super().__init__()
        self.video_expert = video_expert
        self.action_expert = action_expert
        self.mot = mot
        self.vjepa_online_encoder = vjepa_online_encoder
        self.per_view_vjepa_encoder = per_view_vjepa_encoder
        self.jepa_state_expert = jepa_state_expert
        self.jepa_adapter = jepa_adapter
        self.jepa_supervision_config = dict(jepa_supervision_config or {})
        self.jepa_expert_config = dict(jepa_expert_config or {})
        self.jepa_adapter_config = dict(jepa_adapter_config or {})
        raw_branch_config = dict(branch_config or {})
        self.branch_config = {
            "video": bool(raw_branch_config.get("video", "video" in self.mot.mixtures)),
            "jepa": bool(raw_branch_config.get("jepa", "jepa_state" in self.mot.mixtures)),
            "action": bool(raw_branch_config.get("action", "action" in self.mot.mixtures)),
        }
        self.jepa_supervision_enabled = bool(self.jepa_supervision_config.get("enabled", False))
        if self.jepa_supervision_enabled:
            if "jepa_state" not in self.mot.mixtures:
                raise ValueError("Enabled JEPA supervision requires MoT mixture `jepa_state`.")
            missing = [
                name
                for name, module in {
                    "vjepa_online_encoder": self.vjepa_online_encoder,
                    "jepa_state_expert": self.jepa_state_expert,
                }.items()
                if module is None
            ]
            if missing:
                raise ValueError(f"Enabled JEPA supervision missing modules: {missing}")
        # Preserve the registered module alias used by existing checkpoints.
        self.dit = self.mot

        self.vae = vae
        self.text_encoder = text_encoder
        self.tokenizer = tokenizer
        if text_dim is None:
            if self.text_encoder is None:
                raise ValueError("`text_dim` is required when `text_encoder` is not loaded.")
            text_dim = int(self.text_encoder.dim)
        self.text_dim = int(text_dim)
        self.proprio_dim = None if proprio_dim is None else int(proprio_dim)
        if self.proprio_dim is not None:
            proprio_dims = sorted({self.proprio_dim, 8, 14})
            self.proprio_dims = tuple(proprio_dims)
            self.proprio_encoder = nn.ModuleDict(
                {
                    str(dim): nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, self.text_dim))
                    for dim in self.proprio_dims
                }
            ).to(torch_dtype)
        else:
            self.proprio_dims = tuple()
            self.proprio_encoder = None

        self.train_video_scheduler = WanContinuousFlowMatchScheduler(
            num_train_timesteps=video_num_train_timesteps,
            shift=video_train_shift,
        )
        self.infer_video_scheduler = WanContinuousFlowMatchScheduler(
            num_train_timesteps=video_num_train_timesteps,
            shift=video_infer_shift,
        )
        self.train_action_scheduler = WanContinuousFlowMatchScheduler(
            num_train_timesteps=action_num_train_timesteps,
            shift=action_train_shift,
        )
        self.infer_action_scheduler = WanContinuousFlowMatchScheduler(
            num_train_timesteps=action_num_train_timesteps,
            shift=action_infer_shift,
        )
        self.train_jepa_scheduler = WanContinuousFlowMatchScheduler(
            num_train_timesteps=jepa_num_train_timesteps,
            shift=jepa_train_shift,
        )
        self.infer_jepa_scheduler = WanContinuousFlowMatchScheduler(
            num_train_timesteps=jepa_num_train_timesteps,
            shift=jepa_infer_shift,
        )
        # Optional aliases for consistency with Wan22Core naming.
        self.train_scheduler = self.train_video_scheduler
        self.infer_scheduler = self.infer_video_scheduler

        self.device = torch.device(device)
        self.torch_dtype = torch_dtype
        self.loss_lambda_video = float(loss_lambda_video)
        self.loss_lambda_action = float(loss_lambda_action)
        self.loss_lambda_jepa = float(loss_lambda_jepa)
        self.loss_lambda_jepa_feature = float(loss_lambda_jepa_feature)
        self.jepa_current_frame_policy = str(
            self.jepa_supervision_config.get("current_frame_policy", "static_2frame_first")
        )
        self.jepa_future_frame_policy = str(
            self.jepa_supervision_config.get("future_frame_policy", "future_clip_tubelet_aligned")
        )
        self.jepa_num_future_steps = int(self.jepa_supervision_config.get("num_future_steps", 4))
        self.jepa_tubelet_size = int(self.jepa_supervision_config.get("vjepa_tubelet_size", 2))
        self.jepa_feature_loss_type = str(self.jepa_supervision_config.get("feature_loss_type", "smooth_l1"))
        self.jepa_attention_mode = str(
            self.jepa_expert_config.get(
                "attention_mode",
                self.jepa_supervision_config.get("attention_mode", "action_reads_video_jepa"),
            )
        )

        self.to(self.device)
        if self.jepa_adapter is not None:
            self.jepa_adapter.eval()
            self.jepa_adapter.requires_grad_(False)

    @classmethod
    def from_wan22_pretrained(
        cls,
        device: str = "cuda",
        torch_dtype: torch.dtype = torch.bfloat16,
        model_id: str = "Wan-AI/Wan2.2-TI2V-5B",
        tokenizer_model_id: str = "Wan-AI/Wan2.1-T2V-1.3B",
        tokenizer_max_len: int = 512,
        load_text_encoder: bool = True,
        proprio_dim: Optional[int] = None,
        redirect_common_files: bool = True,
        video_dit_config: dict[str, Any] | None = None,
        action_dit_config: dict[str, Any] | None = None,
        action_dit_pretrained_path: str | None = None,
        skip_dit_load_from_pretrain: bool = False,
        mot_checkpoint_mixed_attn: bool = True,
        video_train_shift: float = 5.0,
        video_infer_shift: float = 5.0,
        video_num_train_timesteps: int = 1000,
        action_train_shift: float = 5.0,
        action_infer_shift: float = 5.0,
        action_num_train_timesteps: int = 1000,
        jepa_train_shift: float = 5.0,
        jepa_infer_shift: float = 5.0,
        jepa_num_train_timesteps: int = 1000,
        loss_lambda_video: float = 1.0,
        loss_lambda_action: float = 1.0,
        loss_lambda_jepa: Optional[float] = None,
        loss_lambda_jepa_feature: float = 0.0,
        branches: dict[str, Any] | None = None,
        jepa_supervision: dict[str, Any] | None = None,
        jepa_expert: dict[str, Any] | None = None,
        jepa_adapter: dict[str, Any] | None = None,
    ):
        if video_dit_config is None:
            raise ValueError("`video_dit_config` is required for LeapBotWA.from_wan22_pretrained().")
        if "text_dim" not in video_dit_config:
            raise ValueError("`video_dit_config['text_dim']` is required for LeapBotWA.")

        if branches != {"video": False, "jepa": True, "action": True}:
            raise ValueError("Only the released JEPA/action architecture is supported.")
        jepa_supervision = dict(jepa_supervision or {})
        jepa_expert = dict(jepa_expert or {})
        jepa_adapter = dict(jepa_adapter or {})
        if not jepa_supervision.get("enabled") or not jepa_adapter.get("enabled"):
            raise ValueError("The released model requires both mixed V-JEPA and mixed adapter.")
        video_branch_enabled = False
        action_branch_enabled = True
        jepa_branch_enabled = True
        jepa_enabled = True

        expected_video_num_heads = int(video_dit_config["num_heads"])
        expected_video_attn_head_dim = int(video_dit_config["attn_head_dim"])
        expected_video_num_layers = int(video_dit_config["num_layers"])

        components = load_wan22_ti2v_5b_components(
            device=device,
            torch_dtype=torch_dtype,
            model_id=model_id,
            tokenizer_model_id=tokenizer_model_id,
            tokenizer_max_len=tokenizer_max_len,
            redirect_common_files=redirect_common_files,
            dit_config=video_dit_config,
            skip_dit_load_from_pretrain=skip_dit_load_from_pretrain,
            load_text_encoder=load_text_encoder,
            load_dit=video_branch_enabled,
        )

        video_expert = components.dit
        if video_branch_enabled and video_expert is None:
            raise ValueError("Video branch is enabled but the Wan video DiT was not loaded.")
        action_expert = ActionDiT.from_pretrained(
            action_dit_config=action_dit_config,
            action_dit_pretrained_path=action_dit_pretrained_path,
            skip_dit_load_from_pretrain=skip_dit_load_from_pretrain,
            device=device,
            torch_dtype=torch_dtype,
        )
        if int(action_expert.num_heads) != expected_video_num_heads:
            raise ValueError("ActionDiT `num_heads` must match video_dit_config for MoT mixed attention.")
        if int(action_expert.attn_head_dim) != expected_video_attn_head_dim:
            raise ValueError("ActionDiT `attn_head_dim` must match video_dit_config for MoT mixed attention.")
        if int(len(action_expert.blocks)) != expected_video_num_layers:
            raise ValueError("ActionDiT `num_layers` must match video_dit_config.")

        mixtures: dict[str, nn.Module] = {}
        pass
        vjepa_online_encoder = None
        per_view_vjepa_encoder = None
        jepa_state_expert = None
        frozen_jepa_adapter = None
        effective_jepa_dim = None
        effective_loss_lambda_jepa = 0.0

        jepa_source_config = jepa_supervision
        vjepa_dim = int(jepa_source_config.get("vjepa_dim", 1408))
        vjepa_image_size = int(jepa_source_config.get("vjepa_image_size", 384))
        vjepa_patch_size = int(jepa_source_config.get("vjepa_patch_size", 16))
        vjepa_tubelet_size = int(jepa_source_config.get("vjepa_tubelet_size", 2))
        num_jepa_tokens = jepa_source_config.get("num_jepa_tokens", None)
        if num_jepa_tokens is not None:
            num_jepa_tokens = int(num_jepa_tokens)

        effective_jepa_dim = vjepa_dim
        adapter_checkpoint_path = jepa_adapter.get("checkpoint_path")
        if not adapter_checkpoint_path:
            raise ValueError("`model.jepa_adapter.checkpoint_path` is required when adapter is enabled.")
        frozen_jepa_adapter = FrozenJepaAdapter.from_checkpoint(
            adapter_checkpoint_path,
            input_dim=vjepa_dim,
            device=device,
            torch_dtype=torch_dtype,
            freeze=bool(jepa_adapter.get("freeze", True)),
        )
        effective_jepa_dim = int(frozen_jepa_adapter.latent_dim)
        logger.info(
            "JEPA adapter enabled: raw_vjepa_dim=%d adapter_latent_dim=%d",
            vjepa_dim,
            effective_jepa_dim,
        )

        vjepa_device = jepa_source_config.get("vjepa_device", "same_as_model")
        if vjepa_device in (None, "same_as_model", "model"):
            vjepa_device = device
        vjepa_online_encoder = VJEPAOnlineEncoderWrapper(
            vjepa2_root=str(jepa_source_config.get("vjepa2_root", DEFAULT_VJEPA2_ROOT)),
            checkpoint_path=str(jepa_source_config.get("vjepa_checkpoint", jepa_source_config.get("checkpoint_path", DEFAULT_VJEPA2_CHECKPOINT))),
            image_size=vjepa_image_size,
            patch_size=vjepa_patch_size,
            tubelet_size=vjepa_tubelet_size,
            vjepa_dim=vjepa_dim,
            num_jepa_tokens=num_jepa_tokens,
            dtype=jepa_source_config.get("vjepa_dtype", torch_dtype),
            device=vjepa_device,
        )
        per_view_vjepa_encoder = PerViewVJEPAEncoder(
            vjepa_online_encoder,
            num_view_slots=int(jepa_source_config.get("num_view_slots", 4)),
        )


        jepa_state_vjepa_dim = int(effective_jepa_dim)
        num_future_steps = int(jepa_supervision.get("num_future_steps", 4))
        jepa_hidden_dim = int(jepa_expert.get("hidden_dim", jepa_state_vjepa_dim))
        jepa_num_layers = int(jepa_expert.get("num_layers", expected_video_num_layers))
        jepa_num_heads = int(jepa_expert.get("num_heads", expected_video_num_heads))
        jepa_attn_head_dim = int(jepa_expert.get("attn_head_dim", expected_video_attn_head_dim))
        jepa_ffn_dim = int(jepa_expert.get("ffn_dim", jepa_hidden_dim * 4))
        jepa_state_config = {
            "vjepa_dim": jepa_state_vjepa_dim,
            "hidden_dim": jepa_hidden_dim,
            "ffn_dim": jepa_ffn_dim,
            "text_dim": int(jepa_expert.get("text_dim", video_dit_config["text_dim"])),
            "freq_dim": int(jepa_expert.get("freq_dim", video_dit_config.get("freq_dim", 256))),
            "eps": float(jepa_expert.get("eps", video_dit_config.get("eps", 1e-6))),
            "num_heads": jepa_num_heads,
            "attn_head_dim": jepa_attn_head_dim,
            "num_layers": jepa_num_layers,
            "image_size": vjepa_image_size,
            "patch_size": vjepa_patch_size,
            "num_view_slots": int(jepa_supervision.get("num_view_slots", 4)),
            "use_t_mod": bool(jepa_expert.get("use_t_mod", True)),
            "position_mode": str(jepa_expert.get("position_mode", "vjepa3d_rope")),
            "use_gradient_checkpointing": bool(
                jepa_expert.get("use_gradient_checkpointing", mot_checkpoint_mixed_attn)
            ),
        }
        jepa_state_expert = JEPAMoTStateExpert.from_pretrained(
            jepa_expert_config=jepa_state_config,
            jepa_expert_pretrained_path=(
                jepa_expert.get("pretrained_path")
                or jepa_expert.get("jepa_expert_pretrained_path")
                or jepa_expert.get("backbone_pretrained_path")
            ),
            skip_dit_load_from_pretrain=skip_dit_load_from_pretrain,
            device=device,
            torch_dtype=torch_dtype,
        )
        if int(jepa_state_expert.num_heads) != expected_video_num_heads:
            raise ValueError("JEPA expert `num_heads` must match video_dit_config/action expert for MoT mixed attention.")
        if int(jepa_state_expert.attn_head_dim) != expected_video_attn_head_dim:
            raise ValueError("JEPA expert `attn_head_dim` must match video_dit_config/action expert for MoT mixed attention.")
        if int(len(jepa_state_expert.blocks)) != expected_video_num_layers:
            raise ValueError("JEPA expert `num_layers` must match video_dit_config/action expert.")

        if not bool(jepa_supervision.get("online_encoder", True)):
            raise ValueError("JEPA denoising supervision requires online V-JEPA encoding.")

        mixtures["jepa_state"] = jepa_state_expert
        if loss_lambda_jepa is None:
            raise ValueError("`loss.lambda_jepa` is required when `jepa_supervision.enabled=true`.")
        effective_loss_lambda_jepa = float(loss_lambda_jepa)

        mixtures["action"] = action_expert

        mot = MoT(
            mixtures=mixtures,
            mot_checkpoint_mixed_attn=mot_checkpoint_mixed_attn,
        )

        model = cls(
            video_expert=video_expert if "video" in mixtures else None,
            action_expert=action_expert,
            mot=mot,
            vae=components.vae,
            text_encoder=components.text_encoder,
            tokenizer=components.tokenizer,
            text_dim=int(video_dit_config["text_dim"]),
            proprio_dim=proprio_dim,
            device=device,
            torch_dtype=torch_dtype,
            video_train_shift=video_train_shift,
            video_infer_shift=video_infer_shift,
            video_num_train_timesteps=video_num_train_timesteps,
            action_train_shift=action_train_shift,
            action_infer_shift=action_infer_shift,
            action_num_train_timesteps=action_num_train_timesteps,
            jepa_train_shift=jepa_train_shift,
            jepa_infer_shift=jepa_infer_shift,
            jepa_num_train_timesteps=jepa_num_train_timesteps,
            loss_lambda_video=loss_lambda_video,
            loss_lambda_action=loss_lambda_action,
            loss_lambda_jepa=effective_loss_lambda_jepa,
            loss_lambda_jepa_feature=loss_lambda_jepa_feature,
            vjepa_online_encoder=vjepa_online_encoder,
            per_view_vjepa_encoder=per_view_vjepa_encoder,
            jepa_state_expert=jepa_state_expert,
            jepa_adapter=frozen_jepa_adapter,
            jepa_supervision_config=jepa_supervision,
            jepa_expert_config=jepa_expert,
            jepa_adapter_config=jepa_adapter,
            branch_config={
                "video": video_branch_enabled,
                "jepa": jepa_branch_enabled,
                "action": action_branch_enabled,
            },
        )
        model.model_paths = {
            "branches": dict(model.branch_config),
            "video_dit": components.dit_path,
            "vae": components.vae_path,
            "text_encoder": components.text_encoder_path,
            "tokenizer": components.tokenizer_path,
            "action_dit_backbone": (
                "SKIPPED_PRETRAIN" if skip_dit_load_from_pretrain else action_dit_pretrained_path
            ),
            "vjepa_encoder": (
                None if vjepa_online_encoder is None else str(vjepa_online_encoder.checkpoint_path)
            ),
            "jepa_expert_backbone": (
                jepa_expert.get("pretrained_path")
                or jepa_expert.get("jepa_expert_pretrained_path")
                or jepa_expert.get("backbone_pretrained_path")
            ),
            "jepa_adapter": None if frozen_jepa_adapter is None else str(frozen_jepa_adapter.checkpoint_path),
        }
        return model

    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)
        self.mot.to(*args, **kwargs)
        if self.text_encoder is not None:
            self.text_encoder.to(*args, **kwargs)
        if self.vae is not None:
            self.vae.to(*args, **kwargs)
        if self.vjepa_online_encoder is not None:
            self.vjepa_online_encoder.to(*args, **kwargs)
            self.vjepa_online_encoder.train(False)
        if self.jepa_adapter is not None:
            self.jepa_adapter.to(*args, **kwargs)
            self.jepa_adapter.train(False)
        return self


    @torch.no_grad()
    def encode_prompt(self, prompt: Union[str, Sequence[str]]):
        if self.text_encoder is None or self.tokenizer is None:
            raise ValueError(
                "Prompt encoding requires loaded text encoder/tokenizer. "
                "Set `load_text_encoder=true` or provide precomputed `context/context_mask`."
            )
        ids, mask = self.tokenizer(prompt, return_mask=True, add_special_tokens=True)
        ids = ids.to(self.device)
        mask = mask.to(self.device, dtype=torch.bool)
        prompt_emb = self.text_encoder(ids, mask)
        # FIXME: original implementation's zero padding is visible in cross-attn.
        seq_lens = mask.gt(0).sum(dim=1).long()
        for i, v in enumerate(seq_lens):
            prompt_emb[i, v:] = 0
        mask = torch.ones_like(mask)
        return prompt_emb.to(device=self.device), mask

    def _append_proprio_to_context(
        self,
        context: torch.Tensor,
        context_mask: torch.Tensor,
        proprio: Optional[torch.Tensor],
        action_dim: Optional[int] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.proprio_encoder is None or proprio is None:
            return context, context_mask
        if proprio.ndim != 2:
            raise ValueError(f"`proprio` must be 2D [B, D], got shape {tuple(proprio.shape)}")
        key = str(int(proprio.shape[1]))
        if key not in self.proprio_encoder:
            raise ValueError(
                f"No proprio_encoder for proprio_dim={key}, action_dim={action_dim}; "
                f"expected one of {list(self.proprio_encoder.keys())}."
            )
        proprio_token = self.proprio_encoder[key](
            proprio.to(device=self.device, dtype=context.dtype).unsqueeze(1)
        ).to(dtype=context.dtype)
        inactive_anchor = proprio_token.new_zeros(())
        for inactive_key, encoder in self.proprio_encoder.items():
            if inactive_key == key:
                continue
            for parameter in encoder.parameters():
                inactive_anchor = inactive_anchor + parameter.sum() * 0.0
        proprio_token = proprio_token + inactive_anchor
        proprio_mask = torch.ones((context_mask.shape[0], 1), dtype=torch.bool, device=context_mask.device)
        return (
            torch.cat([context, proprio_token], dim=1),
            torch.cat([context_mask, proprio_mask], dim=1),
        )







    def _is_jepa_enabled(self) -> bool:
        return bool(getattr(self, "jepa_supervision_enabled", False))




    @torch.no_grad()
    def _build_mot_attention_mask_jepa_action(
        self,
        jepa_seq_len: int,
        action_seq_len: int,
        device: torch.device,
        jepa_tokens_per_frame: Optional[int] = None,
    ) -> torch.Tensor:
        if jepa_tokens_per_frame is None:
            if self.jepa_state_expert is not None:
                jepa_tokens_per_frame = int(self.jepa_state_expert.num_jepa_tokens)
            else:
                jepa_tokens_per_frame = int(jepa_seq_len)
        jepa_tokens_per_frame = int(jepa_tokens_per_frame)
        if jepa_tokens_per_frame <= 0:
            raise ValueError(f"`jepa_tokens_per_frame` must be positive, got {jepa_tokens_per_frame}.")
        if jepa_seq_len % jepa_tokens_per_frame != 0:
            raise ValueError(
                "JEPA sequence length must be divisible by tokens_per_frame: "
                f"seq={jepa_seq_len}, tokens_per_frame={jepa_tokens_per_frame}."
            )

        total_seq_len = jepa_seq_len + action_seq_len
        mask = torch.zeros((total_seq_len, total_seq_len), dtype=torch.bool, device=device)
        jepa_start, jepa_end = 0, jepa_seq_len
        action_start, action_end = jepa_end, total_seq_len

        mask[action_start:action_end, action_start:action_end] = True
        mask[action_start:action_end, jepa_start:jepa_start + jepa_tokens_per_frame] = True

        jepa_self_mask = torch.ones((jepa_seq_len, jepa_seq_len), dtype=torch.bool, device=device)
        jepa_self_mask[:jepa_tokens_per_frame, jepa_tokens_per_frame:] = False
        mask[jepa_start:jepa_end, jepa_start:jepa_end] = jepa_self_mask
        return mask


    @torch.no_grad()
    def _prepare_jepa_latents(
        self,
        raw_jepa_latents: torch.Tensor,
        view_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if self.jepa_adapter is None:
            return F.layer_norm(raw_jepa_latents.float(), (raw_jepa_latents.shape[-1],)).to(
                device=self.device, dtype=self.torch_dtype
            )

        params = list(self.jepa_adapter.parameters())
        adapter_dtype = params[0].dtype if params else self.torch_dtype
        features = raw_jepa_latents.to(device=self.device, dtype=adapter_dtype)
        encode_params = inspect.signature(self.jepa_adapter.encode).parameters
        if "view_ids" in encode_params:
            adapted = self.jepa_adapter.encode(features, view_ids=view_ids)
        else:
            adapted = self.jepa_adapter.encode(features)
        return adapted.to(device=self.device, dtype=self.torch_dtype)



    def _encode_jepa_current_from_input_image(self, input_image: torch.Tensor) -> dict[str, Any]:
        if self.vjepa_online_encoder is None:
            raise ValueError("V-JEPA online encoder is not initialized.")
        if input_image.ndim == 4:
            z0_jepa, jepa_grid_size = self.vjepa_online_encoder.encode_current(input_image, return_grid=True)
            view_ids = torch.zeros(int(z0_jepa.shape[1]), device=z0_jepa.device, dtype=torch.long)
            raw_latents = z0_jepa.unsqueeze(1)
        elif input_image.ndim == 5:
            if self.per_view_vjepa_encoder is None:
                raise ValueError("Per-view V-JEPA encoder is not initialized.")
            if input_image.shape[2] != 3:
                raise ValueError(f"Per-view JEPA input must be [B,V,3,H,W], got {tuple(input_image.shape)}.")
            tubelet = int(self.jepa_tubelet_size)
            clip = input_image.unsqueeze(3).repeat(1, 1, 1, tubelet, 1, 1)
            encoded = self.per_view_vjepa_encoder.encode_clip(clip)
            raw_latents = encoded["tokens"]
            view_ids = encoded["view_ids"].to(device=raw_latents.device)
            grid_h, grid_w = encoded["grid_size"]
            num_views = int(encoded.get("num_views", int(input_image.shape[1])))
            jepa_grid_size = (int(grid_h), int(grid_w) * num_views)
        else:
            raise ValueError(
                f"JEPA input image must be [B,3,H,W] or [B,V,3,H,W], got {tuple(input_image.shape)}."
            )
        jepa_latents = self._prepare_jepa_latents(raw_latents, view_ids=view_ids)
        return {
            "latents": jepa_latents,
            "raw_latents": raw_latents,
            "view_ids": view_ids,
            "grid_size": jepa_grid_size,
        }

    def _build_jepa_pre_from_input_image(
        self,
        input_image: torch.Tensor,
        context: torch.Tensor,
        context_mask: torch.Tensor,
    ) -> Optional[dict[str, Any]]:
        if not self._is_jepa_enabled():
            return None
        current = self._encode_jepa_current_from_input_image(input_image)
        jepa_latents = current["latents"]
        timestep_jepa = torch.zeros(
            (jepa_latents.shape[0],), dtype=jepa_latents.dtype, device=self.device
        )
        if self.jepa_state_expert is None:
            raise ValueError("JEPA state expert is not initialized.")
        return self.jepa_state_expert.pre_dit(
            jepa_latents=jepa_latents,
            timestep=timestep_jepa,
            context=context,
            context_mask=context_mask,
            grid_size=current["grid_size"],
            view_ids=current["view_ids"],
        )






    # Legacy video+JEPA+action training path is kept intact but is not called
    # when the MoT mixtures exclude the video expert.



    @torch.no_grad()
    def _predict_action_noise_with_cache(
        self,
        latents_action: torch.Tensor,
        timestep_action: torch.Tensor,
        context: torch.Tensor,
        context_mask: torch.Tensor,
        video_kv_cache: list[dict[str, torch.Tensor]],
        attention_mask: torch.Tensor,
        video_seq_len: int,
        static_kv_caches: Optional[dict[str, list[dict[str, torch.Tensor]]]] = None,
        seq_lens: Optional[dict[str, int]] = None,
    ) -> torch.Tensor:
        action_pre = self.action_expert.pre_dit(
            action_tokens=latents_action,
            timestep=timestep_action,
            context=context,
            context_mask=context_mask,
        )
        action_context_payload = {
            "context": action_pre["context"],
            "mask": action_pre["context_mask"],
        }
        if static_kv_caches is not None:
            action_tokens = self.mot.forward_action_with_static_cache(
                action_tokens=action_pre["tokens"],
                action_freqs=action_pre["freqs"],
                action_t_mod=action_pre["t_mod"],
                action_context_payload=action_context_payload,
                static_kv_caches=static_kv_caches,
                attention_mask=attention_mask,
                seq_lens=dict(seq_lens or {"video": video_seq_len}),
            )
        else:
            action_tokens = self.mot.forward_action_with_video_cache(
                action_tokens=action_pre["tokens"],
                action_freqs=action_pre["freqs"],
                action_t_mod=action_pre["t_mod"],
                action_context_payload=action_context_payload,
                video_kv_cache=video_kv_cache,
                attention_mask=attention_mask,
                video_seq_len=video_seq_len,
            )
        return self.action_expert.post_dit(action_tokens, action_pre)



    @torch.no_grad()
    def infer_action(
        self,
        prompt: Optional[str],
        input_image: torch.Tensor,
        action_horizon: int,
        proprio: Optional[torch.Tensor] = None,
        action_dim: Optional[int] = None,
        context: Optional[torch.Tensor] = None,
        context_mask: Optional[torch.Tensor] = None,
        negative_prompt: Optional[str] = None,
        text_cfg_scale: float = 1.0,
        num_inference_steps: int = 20,
        sigma_shift: Optional[float] = None,
        seed: Optional[int] = None,
        rand_device: str = "cpu",
        tiled: bool = False,
        jepa_input_image: Optional[torch.Tensor] = None,
    ) -> dict[str, Any]:
        self.eval()
        if not self._is_jepa_enabled():
            raise ValueError(
                "`infer_action` without the video branch requires JEPA-state supervision."
            )

        if input_image.ndim == 3:
            input_image = input_image.unsqueeze(0)
        if input_image.ndim != 4 or input_image.shape[0] != 1 or input_image.shape[1] != 3:
            raise ValueError(
                f"`input_image` must have shape [1,3,H,W] or [3,H,W], got {tuple(input_image.shape)}"
            )
        _, _, height, width = input_image.shape
        if height % 16 != 0 or width % 16 != 0:
            raise ValueError(
                f"`input_image` must be resized before infer, expected multiples of 16 but got HxW=({height},{width})"
            )
        if proprio is not None:
            if self.proprio_dim is None:
                raise ValueError("`proprio` was provided but `proprio_dim=None` so `proprio_encoder` is disabled.")
            if proprio.ndim == 1:
                proprio = proprio.unsqueeze(0)
            elif proprio.ndim == 2 and proprio.shape[0] == 1:
                pass
            else:
                raise ValueError(f"`proprio` must be [D] or [1,D], got shape {tuple(proprio.shape)}")
            proprio = proprio.to(device=self.device, dtype=self.torch_dtype)

        active_action_dim = int(self.action_expert.action_dim) if action_dim is None else int(action_dim)
        generator = None if seed is None else torch.Generator(device=rand_device).manual_seed(seed)
        latents_action = torch.randn(
            (1, action_horizon, active_action_dim),
            generator=generator,
            device=rand_device,
            dtype=torch.float32,
        ).to(device=self.device, dtype=self.torch_dtype)

        input_image = input_image.to(device=self.device, dtype=self.torch_dtype)
        first_frame_latents = None
        fuse_flag = False
        pass

        use_prompt = prompt is not None
        use_context = context is not None or context_mask is not None
        if use_prompt and use_context:
            raise ValueError("`prompt` and `context/context_mask` are mutually exclusive.")
        if not use_prompt and not use_context:
            raise ValueError("Either `prompt` or both `context/context_mask` must be provided.")

        if use_prompt:
            context, context_mask = self.encode_prompt(prompt)
        else:
            if context is None or context_mask is None:
                raise ValueError("`context` and `context_mask` must be both provided together.")
            if context.ndim == 2:
                context = context.unsqueeze(0)
            if context_mask.ndim == 1:
                context_mask = context_mask.unsqueeze(0)
            if context.ndim != 3 or context_mask.ndim != 2:
                raise ValueError(
                    f"`context/context_mask` must be [B,L,D]/[B,L], got {tuple(context.shape)} and {tuple(context_mask.shape)}"
                )
            context = context.to(device=self.device, dtype=self.torch_dtype, non_blocking=True)
            context_mask = context_mask.to(device=self.device, dtype=torch.bool, non_blocking=True)
        if proprio is not None:
            context, context_mask = self._append_proprio_to_context(
                context=context,
                context_mask=context_mask,
                proprio=proprio,
                action_dim=action_dim,
            )


        infer_timesteps_action, infer_deltas_action = self.infer_action_scheduler.build_inference_schedule(
            num_inference_steps=num_inference_steps,
            device=self.device,
            dtype=latents_action.dtype,
            shift_override=sigma_shift,
        )
        jepa_source_image = input_image if jepa_input_image is None else jepa_input_image.to(device=self.device, dtype=self.torch_dtype)
        jepa_pre = self._build_jepa_pre_from_input_image(jepa_source_image, context, context_mask)
        if jepa_pre is None:
            raise ValueError("JEPA+action-only inference requires initialized JEPA state tokens.")
        jepa_tokens = jepa_pre["tokens"]
        jepa_seq_len = int(jepa_tokens.shape[1])
        action_seq_len = int(latents_action.shape[1])
        attention_mask = self._build_mot_attention_mask_jepa_action(
            jepa_seq_len=jepa_seq_len,
            action_seq_len=action_seq_len,
            device=jepa_tokens.device,
            jepa_tokens_per_frame=int(jepa_pre["meta"]["tokens_per_frame"]),
        )
        jepa_kv_cache = self.mot.prefill_expert_cache(
            expert_name="jepa_state",
            tokens=jepa_tokens,
            freqs=jepa_pre["freqs"],
            t_mod=jepa_pre["t_mod"],
            context_payload={"context": jepa_pre["context"], "mask": jepa_pre["context_mask"]},
            attention_mask=attention_mask[:jepa_seq_len, :jepa_seq_len],
        )
        static_kv_caches = {"jepa_state": jepa_kv_cache}
        static_seq_lens = {"jepa_state": jepa_seq_len}
        for step_t_action, step_delta_action in zip(infer_timesteps_action, infer_deltas_action):
            timestep_action = step_t_action.unsqueeze(0).to(dtype=latents_action.dtype, device=self.device)
            pred_action = self._predict_action_noise_with_cache(
                latents_action=latents_action,
                timestep_action=timestep_action,
                context=context,
                context_mask=context_mask,
                video_kv_cache=[],
                attention_mask=attention_mask,
                video_seq_len=0,
                static_kv_caches=static_kv_caches,
                seq_lens=static_seq_lens,
            )
            latents_action = self.infer_action_scheduler.step(pred_action, step_delta_action, latents_action)

        return {
            "action": latents_action[0].detach().to(device="cpu", dtype=torch.float32),
        }

    def load_checkpoint(self, path):
        payload = torch.load(path, map_location="cpu", weights_only=True)
        if payload.get("branches", self.branch_config) != self.branch_config:
            raise ValueError("Checkpoint architecture does not match the released model")
        if payload.get("mot_expert_order", list(self.mot.expert_order)) != list(self.mot.expert_order):
            raise ValueError("Checkpoint expert order does not match the released model")
        if "mot" not in payload or "proprio_encoder" not in payload:
            raise ValueError("A complete released policy checkpoint is required")
        state = payload["mot"]
        self.mot.load_state_dict(state, strict=True)
        self.proprio_encoder.load_state_dict(payload["proprio_encoder"], strict=True)
        logger.info("Loaded complete policy checkpoint: %s (step=%s)", path, payload.get("step"))
        return payload
