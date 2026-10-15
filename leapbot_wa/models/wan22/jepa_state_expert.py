from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import torch.nn as nn

from leapbot_wa.utils.logging_config import get_logger

from .helpers.gradient import gradient_checkpoint_forward
from .wan_video_dit import DiTBlock, precompute_freqs_cis_3d, sinusoidal_embedding_1d

logger = get_logger(__name__)


class JEPADenoiseHead(nn.Module):
    """Timestep-modulated readout for V-JEPA feature velocity prediction."""

    def __init__(self, hidden_dim: int, out_dim: int, eps: float):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim, eps=eps, elementwise_affine=False)
        self.proj = nn.Linear(hidden_dim, out_dim)
        self.modulation = nn.Parameter(torch.randn(1, 2, hidden_dim) / hidden_dim**0.5)

    def forward(self, tokens: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        if t.ndim == 3:
            shift, scale = (
                self.modulation.unsqueeze(0).to(dtype=t.dtype, device=t.device) + t.unsqueeze(2)
            ).chunk(2, dim=2)
            return self.proj(self.norm(tokens) * (1 + scale.squeeze(2)) + shift.squeeze(2))
        if t.ndim == 2:
            shift, scale = (
                self.modulation.to(dtype=t.dtype, device=t.device) + t.unsqueeze(1)
            ).chunk(2, dim=1)
            shift = shift.squeeze(1).unsqueeze(1)
            scale = scale.squeeze(1).unsqueeze(1)
            return self.proj(self.norm(tokens) * (1 + scale) + shift)
        raise ValueError(f"JEPA head timestep embedding must be [B,S,D] or [B,D], got {tuple(t.shape)}.")


class JEPAMoTStateExpert(nn.Module):
    """MoT-compatible denoising expert for full V-JEPA patch-token latents."""

    JEPA_BACKBONE_SKIP_PREFIXES = (
        "input_norm.",
        "input_proj.",
        "view_residual_embedding.",
        "text_embedding.",
        "time_embedding.",
        "time_projection.",
        "head.",
    )
    JEPA_BACKBONE_META_KEYS = (
        "vjepa_dim",
        "hidden_dim",
        "ffn_dim",
        "num_layers",
        "num_heads",
        "attn_head_dim",
        "eps",
        "image_size",
        "patch_size",
    )

    def __init__(
        self,
        *,
        vjepa_dim: int = 1408,
        hidden_dim: int = 1408,
        ffn_dim: Optional[int] = None,
        text_dim: int = 4096,
        freq_dim: int = 256,
        eps: float = 1e-6,
        num_heads: int = 24,
        attn_head_dim: int = 128,
        num_layers: int = 30,
        image_size: int = 384,
        patch_size: int = 16,
        num_view_slots: int = 1,
        use_t_mod: bool = True,
        position_mode: str = "vjepa3d_rope",
        use_gradient_checkpointing: bool = False,
    ):
        super().__init__()
        self.vjepa_dim = int(vjepa_dim)
        self.hidden_dim = int(hidden_dim)
        self.ffn_dim = int(ffn_dim) if ffn_dim is not None else int(hidden_dim) * 4
        self.text_dim = int(text_dim)
        self.freq_dim = int(freq_dim)
        self.eps = float(eps)
        self.num_heads = int(num_heads)
        self.attn_head_dim = int(attn_head_dim)
        self.num_layers = int(num_layers)
        self.image_size = int(image_size)
        self.patch_size = int(patch_size)
        self.num_view_slots = int(num_view_slots)
        self.use_t_mod = bool(use_t_mod)
        self.position_mode = str(position_mode)
        self.use_gradient_checkpointing = bool(use_gradient_checkpointing)

        if not self.use_t_mod:
            raise ValueError("JEPAMoTStateExpert denoising mode requires `use_t_mod=True`.")
        if self.position_mode != "vjepa3d_rope":
            raise ValueError(f"Unsupported JEPA position_mode: {self.position_mode}")
        if self.vjepa_dim <= 0 or self.hidden_dim <= 0 or self.ffn_dim <= 0:
            raise ValueError("JEPA expert dims must be positive.")
        if self.text_dim <= 0 or self.freq_dim <= 0:
            raise ValueError("JEPA expert text/frequency dims must be positive.")
        if self.num_heads <= 0:
            raise ValueError(f"`num_heads` must be > 0, got {self.num_heads}.")
        if self.attn_head_dim <= 0 or self.attn_head_dim % 2 != 0:
            raise ValueError(
                f"`attn_head_dim` must be positive and even for RoPE, got {self.attn_head_dim}."
            )
        if self.num_layers <= 0:
            raise ValueError(f"`num_layers` must be > 0, got {self.num_layers}.")
        if self.image_size <= 0 or self.patch_size <= 0 or self.image_size % self.patch_size != 0:
            raise ValueError(
                f"`image_size` must be divisible by `patch_size`, got {self.image_size}/{self.patch_size}."
            )
        if self.num_view_slots <= 0:
            raise ValueError(f"`num_view_slots` must be positive, got {self.num_view_slots}.")

        self.grid_h = self.image_size // self.patch_size
        self.grid_w = self.image_size // self.patch_size
        self.num_jepa_tokens = self.grid_h * self.grid_w

        self.input_norm = nn.LayerNorm(self.vjepa_dim, eps=self.eps)
        self.input_proj = nn.Linear(self.vjepa_dim, self.hidden_dim)
        # This conditions only the JEPA DiT and does not modify adapter latents.
        self.view_residual_embedding = nn.Embedding(self.num_view_slots, self.hidden_dim)
        nn.init.zeros_(self.view_residual_embedding.weight)
        self.text_embedding = nn.Sequential(
            nn.Linear(self.text_dim, self.hidden_dim),
            nn.GELU(approximate="tanh"),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )
        self.time_embedding = nn.Sequential(
            nn.Linear(self.freq_dim, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )
        self.time_projection = nn.Sequential(nn.SiLU(), nn.Linear(self.hidden_dim, self.hidden_dim * 6))
        self.blocks = nn.ModuleList(
            [
                DiTBlock(
                    hidden_dim=self.hidden_dim,
                    attn_head_dim=self.attn_head_dim,
                    num_heads=self.num_heads,
                    ffn_dim=self.ffn_dim,
                    eps=self.eps,
                )
                for _ in range(self.num_layers)
            ]
        )
        self.head = JEPADenoiseHead(self.hidden_dim, self.vjepa_dim, self.eps)
        self.freqs = precompute_freqs_cis_3d(self.attn_head_dim)

    @classmethod
    def backbone_key_set(cls, keys) -> set[str]:
        return {
            key
            for key in keys
            if not any(key.startswith(prefix) for prefix in cls.JEPA_BACKBONE_SKIP_PREFIXES)
        }

    @classmethod
    def from_pretrained(
        cls,
        jepa_expert_config: dict[str, Any],
        jepa_expert_pretrained_path: Optional[str] = None,
        skip_dit_load_from_pretrain: bool = False,
        device: str = "cuda",
        torch_dtype: torch.dtype = torch.bfloat16,
    ) -> "JEPAMoTStateExpert":
        expert = cls(**jepa_expert_config).to(device=device, dtype=torch_dtype)
        if skip_dit_load_from_pretrain:
            logger.info(
                "Skipping JEPAMoTStateExpert pretrained load (`skip_dit_load_from_pretrain=True`)."
            )
            return expert
        if not jepa_expert_pretrained_path:
            logger.info("No `jepa_expert_pretrained_path` provided; initializing JEPA expert randomly.")
            return expert

        path = Path(jepa_expert_pretrained_path)
        if not path.is_absolute():
            path = Path(__file__).resolve().parents[4] / path
        if not os.path.isfile(path):
            raise FileNotFoundError(f"`jepa_expert_pretrained_path` does not exist: {path}")

        state = expert.state_dict()
        expected_keys = cls.backbone_key_set(state.keys())
        payload = torch.load(str(path), map_location="cpu")
        if not isinstance(payload, dict):
            raise ValueError(f"Invalid JEPA backbone payload type from {path}: {type(payload)}.")
        policy = payload.get("policy", {})
        if policy:
            logger.info("JEPAMoTStateExpert backbone payload policy: %s", policy)

        meta = payload.get("meta")
        if not isinstance(meta, dict):
            raise ValueError(f"`meta` must be a dict in {path}.")
        expected_meta = {
            "vjepa_dim": int(jepa_expert_config["vjepa_dim"]),
            "hidden_dim": int(jepa_expert_config["hidden_dim"]),
            "ffn_dim": int(jepa_expert_config["ffn_dim"]),
            "num_layers": int(jepa_expert_config["num_layers"]),
            "num_heads": int(jepa_expert_config["num_heads"]),
            "attn_head_dim": int(jepa_expert_config["attn_head_dim"]),
            "eps": float(jepa_expert_config["eps"]),
            "image_size": int(jepa_expert_config["image_size"]),
            "patch_size": int(jepa_expert_config["patch_size"]),
        }
        for key in cls.JEPA_BACKBONE_META_KEYS:
            if key not in meta:
                raise ValueError(f"`meta.{key}` missing in {path}.")
            if key == "eps":
                if abs(float(meta[key]) - float(expected_meta[key])) > 1e-12:
                    raise ValueError(f"`meta.{key}` mismatch in {path}: expected {expected_meta[key]}, got {meta[key]}.")
            elif int(meta[key]) != int(expected_meta[key]):
                raise ValueError(f"`meta.{key}` mismatch in {path}: expected {expected_meta[key]}, got {meta[key]}.")

        backbone_state_dict = payload.get("backbone_state_dict")
        if not isinstance(backbone_state_dict, dict):
            raise ValueError(f"`backbone_state_dict` must be a dict in {path}.")
        provided_keys = set(backbone_state_dict.keys())
        missing_keys = sorted(expected_keys - provided_keys)
        unexpected_keys = sorted(provided_keys - expected_keys)
        if missing_keys or unexpected_keys:
            raise ValueError(
                "JEPA backbone key mismatch in preprocessed payload. "
                f"missing={missing_keys[:10]}{'...' if len(missing_keys) > 10 else ''}, "
                f"unexpected={unexpected_keys[:10]}{'...' if len(unexpected_keys) > 10 else ''}"
            )

        merged_state = dict(state)
        for key in expected_keys:
            value = backbone_state_dict[key]
            if not isinstance(value, torch.Tensor):
                raise ValueError(f"`backbone_state_dict[{key}]` must be a tensor in {path}.")
            target = merged_state[key]
            if tuple(value.shape) != tuple(target.shape):
                raise ValueError(
                    f"Shape mismatch for `{key}` in {path}: expected {tuple(target.shape)}, got {tuple(value.shape)}."
                )
            merged_state[key] = value.to(device=target.device, dtype=target.dtype)

        expert.load_state_dict(merged_state, strict=True)
        logger.info(
            "Loaded JEPAMoTStateExpert backbone from %s (keys=%d; random_kept_prefixes=%s).",
            path,
            len(expected_keys),
            list(cls.JEPA_BACKBONE_SKIP_PREFIXES),
        )
        return expert.to(device=device, dtype=torch_dtype)

    def _resolve_grid_size(self, grid_size: Optional[tuple[int, int]]) -> tuple[int, int]:
        if grid_size is None:
            return self.grid_h, self.grid_w
        if len(grid_size) != 2:
            raise ValueError(f"`grid_size` must be (grid_h, grid_w), got {grid_size}.")
        h, w = int(grid_size[0]), int(grid_size[1])
        if h <= 0 or w <= 0:
            raise ValueError(f"JEPA spatial grid must be positive, got {(h, w)}.")
        return h, w

    def _build_freqs(
        self,
        *,
        num_frames: int,
        grid_size: Optional[tuple[int, int]],
        device: torch.device,
    ) -> torch.Tensor:
        f = int(num_frames)
        h, w = self._resolve_grid_size(grid_size)
        if f <= 0:
            raise ValueError(f"JEPA temporal grid must be positive, got {f}.")
        max_f, max_h, max_w = self.freqs[0].shape[0], self.freqs[1].shape[0], self.freqs[2].shape[0]
        if f > max_f or h > max_h or w > max_w:
            raise ValueError(
                "JEPA RoPE grid exceeds precomputed frequency table: "
                f"requested={(f, h, w)}, available={(max_f, max_h, max_w)}."
            )
        tokens_per_frame = h * w
        return torch.cat(
            [
                self.freqs[0][:f].view(f, 1, 1, -1).expand(f, h, w, -1),
                self.freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
                self.freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1),
            ],
            dim=-1,
        ).reshape(f * tokens_per_frame, 1, -1).to(device)

    def pre_dit(
        self,
        *,
        jepa_latents: torch.Tensor,
        timestep: torch.Tensor,
        context: torch.Tensor,
        context_mask: Optional[torch.Tensor] = None,
        grid_size: Optional[tuple[int, int]] = None,
        view_ids: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        if jepa_latents.ndim != 4:
            raise ValueError(f"`jepa_latents` must be [B, F, N, Dv], got {tuple(jepa_latents.shape)}.")
        grid_h, grid_w = self._resolve_grid_size(grid_size)
        tokens_per_frame = grid_h * grid_w
        if jepa_latents.shape[2] != tokens_per_frame:
            raise ValueError(
                "`jepa_latents` must keep full V-JEPA spatial tokens for the active grid: "
                f"grid={(grid_h, grid_w)} expected N={tokens_per_frame}, got {jepa_latents.shape[2]}."
            )
        if jepa_latents.shape[3] != self.vjepa_dim:
            raise ValueError(
                f"`jepa_latents` last dim must be {self.vjepa_dim}, got {jepa_latents.shape[3]}."
            )
        if timestep.ndim != 1:
            raise ValueError(f"`timestep` must be 1D [B] or [1], got {tuple(timestep.shape)}.")
        if context.ndim != 3:
            raise ValueError(f"`context` must be [B, L, D], got {tuple(context.shape)}.")
        batch_size, num_frames = int(jepa_latents.shape[0]), int(jepa_latents.shape[1])
        if context.shape[0] != batch_size:
            raise ValueError(
                f"Batch mismatch between JEPA tokens and context: {batch_size} vs {context.shape[0]}."
            )
        if context.shape[-1] != self.text_dim:
            raise ValueError(
                f"JEPA context last dim must be raw text_dim={self.text_dim}, got {context.shape[-1]}."
            )
        if timestep.shape[0] not in (1, batch_size):
            raise ValueError(f"`timestep` length must be 1 or batch_size({batch_size}), got {timestep.shape[0]}.")
        if timestep.shape[0] == 1 and batch_size > 1:
            if self.training:
                raise ValueError("During training, JEPA timestep length must match batch_size.")
            timestep = timestep.expand(batch_size)
        if context_mask is None:
            context_mask = torch.ones((batch_size, context.shape[1]), dtype=torch.bool, device=context.device)
        elif context_mask.ndim != 2:
            raise ValueError(f"`context_mask` must be [B, L], got {tuple(context_mask.shape)}.")
        elif context_mask.shape[0] != batch_size or context_mask.shape[1] != context.shape[1]:
            raise ValueError(
                f"`context_mask` shape mismatch: got {tuple(context_mask.shape)} vs context {tuple(context.shape)}."
            )

        target_dtype = self.input_proj.weight.dtype
        seq_len = num_frames * tokens_per_frame
        flat_latents = jepa_latents.reshape(batch_size, seq_len, self.vjepa_dim)
        tokens = self.input_proj(self.input_norm(flat_latents.to(device=context.device, dtype=target_dtype)))
        if view_ids is None:
            view_ids = torch.zeros(tokens_per_frame, device=tokens.device, dtype=torch.long)
        if view_ids.ndim != 1 or int(view_ids.shape[0]) != tokens_per_frame:
            raise ValueError(
                f"`view_ids` must be 1D with length N={tokens_per_frame}, got {tuple(view_ids.shape)}."
            )
        view_ids = view_ids.to(device=tokens.device, dtype=torch.long)
        if view_ids.numel() > 0:
            min_view_id = int(view_ids.min().item())
            max_view_id = int(view_ids.max().item())
            if min_view_id < 0 or max_view_id >= self.num_view_slots:
                raise ValueError(
                    f"`view_ids` must be in [0,{self.num_view_slots - 1}], "
                    f"got min={min_view_id}, max={max_view_id}."
                )
        view_residual = self.view_residual_embedding(view_ids).to(dtype=tokens.dtype)
        view_residual = view_residual.repeat(num_frames, 1).unsqueeze(0)
        tokens = tokens + view_residual

        token_timesteps = torch.ones(
            (batch_size, num_frames, tokens_per_frame),
            dtype=timestep.dtype,
            device=timestep.device,
        ) * timestep.view(batch_size, 1, 1)
        token_timesteps[:, 0, :] = 0
        token_t_emb = sinusoidal_embedding_1d(self.freq_dim, token_timesteps.reshape(-1))
        t = self.time_embedding(token_t_emb).reshape(batch_size, seq_len, self.hidden_dim)
        t_mod = self.time_projection(t).unflatten(2, (6, self.hidden_dim))

        context_emb = self.text_embedding(context.to(device=tokens.device, dtype=target_dtype))
        context_attn_mask = context_mask.to(device=tokens.device, dtype=torch.bool).unsqueeze(1).expand(
            -1, seq_len, -1
        )
        freqs = self._build_freqs(num_frames=num_frames, grid_size=(grid_h, grid_w), device=tokens.device)
        return {
            "tokens": tokens,
            "freqs": freqs,
            "t": t,
            "t_mod": t_mod,
            "context": context_emb,
            "context_mask": context_attn_mask,
            "meta": {
                "batch_size": batch_size,
                "grid_size": (num_frames, grid_h, grid_w),
                "tokens_per_frame": tokens_per_frame,
                "position_mode": self.position_mode,
            },
        }

    def post_dit(self, tokens: torch.Tensor, pre_state: Dict[str, Any]) -> torch.Tensor:
        f, grid_h, grid_w = pre_state["meta"]["grid_size"]
        batch_size = int(pre_state["meta"]["batch_size"])
        tokens_per_frame = int(pre_state["meta"]["tokens_per_frame"])
        expected_seq_len = int(f) * tokens_per_frame
        if tokens.ndim != 3 or tokens.shape[0] != batch_size or tokens.shape[1] != expected_seq_len:
            raise ValueError(
                "JEPA post_dit token shape mismatch: "
                f"got {tuple(tokens.shape)}, expected ({batch_size}, {expected_seq_len}, {self.hidden_dim})."
            )
        if tokens_per_frame != int(grid_h) * int(grid_w):
            raise ValueError(
                "JEPA post_dit grid/token metadata mismatch: "
                f"grid={(grid_h, grid_w)}, tokens_per_frame={tokens_per_frame}."
            )
        pred = self.head(tokens, pre_state["t"])
        return pred.reshape(batch_size, int(f), tokens_per_frame, self.vjepa_dim)

    def forward(
        self,
        jepa_latents: torch.Tensor,
        timestep: torch.Tensor,
        context: torch.Tensor,
        context_mask: Optional[torch.Tensor] = None,
        grid_size: Optional[tuple[int, int]] = None,
        view_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        pre_state = self.pre_dit(
            jepa_latents=jepa_latents,
            timestep=timestep,
            context=context,
            context_mask=context_mask,
            grid_size=grid_size,
            view_ids=view_ids,
        )
        x = pre_state["tokens"]
        context_emb = pre_state["context"]
        t_mod = pre_state["t_mod"]
        freqs = pre_state["freqs"]
        context_mask = pre_state["context_mask"]

        for block in self.blocks:
            if self.use_gradient_checkpointing:
                x = gradient_checkpoint_forward(
                    block,
                    self.use_gradient_checkpointing,
                    x,
                    context_emb,
                    t_mod,
                    freqs,
                    context_mask=context_mask,
                )
            else:
                x = block(x, context_emb, t_mod, freqs, context_mask=context_mask)
        return self.post_dit(x, pre_state)
