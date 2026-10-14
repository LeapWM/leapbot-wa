from __future__ import annotations

import math
import os
from dataclasses import dataclass
from pathlib import Path
from collections.abc import Sequence
from typing import Any, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from leapbot_wa.utils.logging_config import get_logger

try:
    from transformers.activations import ACT2FN
except ImportError:  # pragma: no cover - transformers is already a project dependency.
    ACT2FN = {"gelu": F.gelu}

logger = get_logger(__name__)


@dataclass
class ViTMAEConfig:
    hidden_size: int = 768
    num_hidden_layers: int = 12
    num_attention_heads: int = 12
    intermediate_size: int = 3072
    hidden_act: str = "gelu"
    hidden_dropout_prob: float = 0.0
    attention_probs_dropout_prob: float = 0.0
    layer_norm_eps: float = 1e-12
    qkv_bias: bool = True


class ViTMAESelfAttention(nn.Module):
    def __init__(self, config: ViTMAEConfig) -> None:
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                f"hidden_size={config.hidden_size} must be divisible by "
                f"num_attention_heads={config.num_attention_heads}."
            )
        self.num_attention_heads = int(config.num_attention_heads)
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.query = nn.Linear(config.hidden_size, self.all_head_size, bias=config.qkv_bias)
        self.key = nn.Linear(config.hidden_size, self.all_head_size, bias=config.qkv_bias)
        self.value = nn.Linear(config.hidden_size, self.all_head_size, bias=config.qkv_bias)
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(
        self,
        hidden_states: torch.Tensor,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ) -> Union[Tuple[torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]:
        mixed_query_layer = self.query(hidden_states)
        key_layer = self.transpose_for_scores(self.key(hidden_states))
        value_layer = self.transpose_for_scores(self.value(hidden_states))
        query_layer = self.transpose_for_scores(mixed_query_layer)
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)
        attention_probs = self.dropout(attention_probs)
        if head_mask is not None:
            attention_probs = attention_probs * head_mask
        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(new_context_layer_shape)
        if output_attentions:
            return context_layer, attention_probs
        return (context_layer,)


class ViTMAESelfOutput(nn.Module):
    def __init__(self, config: ViTMAEConfig) -> None:
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        del input_tensor
        hidden_states = self.dense(hidden_states)
        return self.dropout(hidden_states)


class ViTMAEAttention(nn.Module):
    def __init__(self, config: ViTMAEConfig) -> None:
        super().__init__()
        self.attention = ViTMAESelfAttention(config)
        self.output = ViTMAESelfOutput(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ) -> Union[Tuple[torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]:
        self_outputs = self.attention(hidden_states, head_mask, output_attentions)
        attention_output = self.output(self_outputs[0], hidden_states)
        return (attention_output,) + self_outputs[1:]


class ViTMAEIntermediate(nn.Module):
    def __init__(self, config: ViTMAEConfig) -> None:
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        self.intermediate_act_fn = ACT2FN[config.hidden_act] if isinstance(config.hidden_act, str) else config.hidden_act

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.intermediate_act_fn(self.dense(hidden_states))


class ViTMAEOutput(nn.Module):
    def __init__(self, config: ViTMAEConfig) -> None:
        super().__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        return hidden_states + input_tensor


class ViTMAELayer(nn.Module):
    def __init__(self, config: ViTMAEConfig) -> None:
        super().__init__()
        self.attention = ViTMAEAttention(config)
        self.intermediate = ViTMAEIntermediate(config)
        self.output = ViTMAEOutput(config)
        self.layernorm_before = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.layernorm_after = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ) -> Union[Tuple[torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]:
        self_attention_outputs = self.attention(
            self.layernorm_before(hidden_states),
            head_mask,
            output_attentions=output_attentions,
        )
        attention_output = self_attention_outputs[0]
        outputs = self_attention_outputs[1:]
        hidden_states = attention_output + hidden_states
        layer_output = self.layernorm_after(hidden_states)
        layer_output = self.intermediate(layer_output)
        layer_output = self.output(layer_output, hidden_states)
        return (layer_output,) + outputs


class DiagonalGaussian(nn.Module):
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = x.chunk(2, dim=-1)
        if self.training:
            std = (0.5 * logvar).exp()
            z = mu + std * torch.randn_like(std)
        else:
            z = mu
        return z, mu, logvar


def _best_head_count(dim: int, max_heads: int = 16) -> int:
    for h in range(min(max_heads, dim), 0, -1):
        if dim % h == 0:
            return h
    return 1


class SVAEAdapter(nn.Module):
    def __init__(
        self,
        input_dim: int,
        latent_dim: int = 96,
        num_heads: int = 12,
        num_layers: int = 3,
        intermediate_size: int = 3072,
        progressive: bool = False,
        mid_dim: Optional[int] = None,
        mid_heads: Optional[int] = None,
        latent_layers: int = 0,
        latent_heads: Optional[int] = None,
    ) -> None:
        super().__init__()
        self._input_dim = int(input_dim)
        self._latent_dim = int(latent_dim)
        self._progressive = bool(progressive)
        self._latent_layers = int(latent_layers)
        block_cfg = ViTMAEConfig(
            hidden_size=self._input_dim,
            num_hidden_layers=int(num_layers),
            num_attention_heads=int(num_heads),
            intermediate_size=int(intermediate_size),
        )
        if self._progressive:
            enc_layers_high = max(int(num_layers) - 1, 1)
            enc_layers_mid = max(int(num_layers) - enc_layers_high, 1)
            if mid_dim is None:
                mid_dim = int((self._input_dim * self._latent_dim) ** 0.5)
            if mid_heads is None:
                mid_heads = _best_head_count(int(mid_dim), max_heads=16)
            mid_dim = (int(mid_dim) // int(mid_heads)) * int(mid_heads)
            self._mid_dim = mid_dim
            mid_cfg = ViTMAEConfig(
                hidden_size=mid_dim,
                num_hidden_layers=enc_layers_mid,
                num_attention_heads=int(mid_heads),
                intermediate_size=mid_dim * 4,
            )
            self.enc_blocks = nn.ModuleList([ViTMAELayer(block_cfg) for _ in range(enc_layers_high)])
            self.enc_norm_high = nn.LayerNorm(self._input_dim)
            self.enc_down = nn.Linear(self._input_dim, mid_dim)
            self.enc_blocks_mid = nn.ModuleList([ViTMAELayer(mid_cfg) for _ in range(enc_layers_mid)])
            self.enc_norm = nn.LayerNorm(mid_dim)
            self.enc_proj = nn.Linear(mid_dim, self._latent_dim * 2)
            self.dec_proj = nn.Linear(self._latent_dim, mid_dim)
            self.dec_blocks_mid = nn.ModuleList([ViTMAELayer(mid_cfg) for _ in range(enc_layers_mid)])
            self.dec_norm_mid = nn.LayerNorm(mid_dim)
            self.dec_up = nn.Linear(mid_dim, self._input_dim)
            self.dec_blocks = nn.ModuleList([ViTMAELayer(block_cfg) for _ in range(enc_layers_high)])
            self.dec_norm = nn.LayerNorm(self._input_dim)
        else:
            self.enc_blocks = nn.ModuleList([ViTMAELayer(block_cfg) for _ in range(int(num_layers))])
            self.enc_norm = nn.LayerNorm(self._input_dim)
            self.enc_proj = nn.Linear(self._input_dim, self._latent_dim * 2)
            self.dec_proj = nn.Linear(self._latent_dim, self._input_dim)
            self.dec_blocks = nn.ModuleList([ViTMAELayer(block_cfg) for _ in range(int(num_layers))])
            self.dec_norm = nn.LayerNorm(self._input_dim)
        self.gaussian = DiagonalGaussian()
        if self._latent_layers > 0:
            if latent_heads is None:
                latent_heads = _best_head_count(self._latent_dim, max_heads=8)
            lat_cfg = ViTMAEConfig(
                hidden_size=self._latent_dim,
                num_hidden_layers=self._latent_layers,
                num_attention_heads=int(latent_heads),
                intermediate_size=self._latent_dim * 4,
            )
            self.latent_enc_blocks = nn.ModuleList([ViTMAELayer(lat_cfg) for _ in range(self._latent_layers)])
            self.latent_enc_norm = nn.LayerNorm(self._latent_dim)
            self.latent_dec_blocks = nn.ModuleList([ViTMAELayer(lat_cfg) for _ in range(self._latent_layers)])
            self.latent_dec_norm = nn.LayerNorm(self._latent_dim)

    @property
    def latent_dim(self) -> int:
        return self._latent_dim

    @property
    def input_dim(self) -> int:
        return self._input_dim

    @staticmethod
    def _to_seq(z: torch.Tensor):
        if z.dim() == 5:
            b, t, h, w, c = z.shape
            return z.reshape(b * t, h * w, c), (b, t, h, w, c), True
        if z.dim() != 4:
            raise ValueError(f"SVAEAdapter expects [B,T,N,C] or [B,T,H,W,C], got {tuple(z.shape)}.")
        b, t, n, c = z.shape
        return z.reshape(b * t, n, c), (b, t, n, c), False

    @staticmethod
    def _from_seq(z: torch.Tensor, meta, is_spatial: bool, new_c: int):
        if is_spatial:
            b, t, h, w, _ = meta
            return z.reshape(b, t, h, w, new_c)
        b, t, n, _ = meta
        return z.reshape(b, t, n, new_c)

    def encode(self, z: torch.Tensor) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        x, meta, is_spatial = self._to_seq(z)
        for blk in self.enc_blocks:
            x = blk(x)[0]
        if self._progressive:
            x = self.enc_norm_high(x)
            x = self.enc_down(x)
            for blk in self.enc_blocks_mid:
                x = blk(x)[0]
        x = self.enc_norm(x)
        x = self.enc_proj(x)
        z_l, mu, logvar = self.gaussian(x)
        if self._latent_layers > 0:
            z_l = self._refine_latent_enc(z_l)
        z_l = self._from_seq(z_l, meta, is_spatial, self._latent_dim)
        if self.training:
            mu = self._from_seq(mu, meta, is_spatial, self._latent_dim)
            logvar = self._from_seq(logvar, meta, is_spatial, self._latent_dim)
            return z_l, mu, logvar
        return z_l

    def _refine_latent_enc(self, z_l: torch.Tensor) -> torch.Tensor:
        for blk in self.latent_enc_blocks:
            z_l = blk(z_l)[0]
        return self.latent_enc_norm(z_l)

    def _refine_latent_dec(self, z_l: torch.Tensor) -> torch.Tensor:
        for blk in self.latent_dec_blocks:
            z_l = blk(z_l)[0]
        return self.latent_dec_norm(z_l)

    def decode(self, z_l: torch.Tensor) -> torch.Tensor:
        x, meta, is_spatial = self._to_seq(z_l)
        if self._latent_layers > 0:
            x = self._refine_latent_dec(x)
        x = self.dec_proj(x)
        if self._progressive:
            for blk in self.dec_blocks_mid:
                x = blk(x)[0]
            x = self.dec_norm_mid(x)
            x = self.dec_up(x)
        for blk in self.dec_blocks:
            x = blk(x)[0]
        x = self.dec_norm(x)
        return self._from_seq(x, meta, is_spatial, self._input_dim)

    def forward(
        self,
        features: torch.Tensor,
        *,
        return_reconstruction: bool = False,
    ):
        """Encode shared V-JEPA features without conditioning on camera identity."""
        enc_out = self.encode(features)
        if not return_reconstruction:
            return enc_out
        z_l = enc_out[0] if isinstance(enc_out, tuple) else enc_out
        return enc_out, self.decode(z_l)


class ViewAwareSVAEAdapter(nn.Module):
    """SVAE adapter with a learned view embedding added before encoding."""

    def __init__(
        self,
        adapter: SVAEAdapter,
        *,
        num_view_slots: int = 2,
        view_schema: Optional[Sequence[str]] = None,
    ) -> None:
        super().__init__()
        self.adapter = adapter
        self.num_view_slots = int(num_view_slots)
        if self.num_view_slots <= 0:
            raise ValueError(f"`num_view_slots` must be positive, got {self.num_view_slots}.")
        self.view_embedding = nn.Embedding(self.num_view_slots, adapter.input_dim)
        self.view_schema = list(view_schema or [f"view_{i}" for i in range(self.num_view_slots)])
        if len(self.view_schema) != self.num_view_slots:
            raise ValueError(
                f"`view_schema` length must equal num_view_slots={self.num_view_slots}, got {len(self.view_schema)}."
            )

    @property
    def input_dim(self) -> int:
        return self.adapter.input_dim

    @property
    def latent_dim(self) -> int:
        return self.adapter.latent_dim

    def _apply_view_embedding(self, features: torch.Tensor, view_ids: torch.Tensor) -> torch.Tensor:
        if features.ndim != 4:
            raise ValueError(f"View-aware adapter expects features [B,T,N,C], got {tuple(features.shape)}.")
        if view_ids.ndim != 1:
            raise ValueError(f"`view_ids` must be 1D [N], got {tuple(view_ids.shape)}.")
        if int(view_ids.shape[0]) != int(features.shape[2]):
            raise ValueError(
                f"`view_ids` length must match token count N={features.shape[2]}, got {view_ids.shape[0]}."
            )
        view_ids = view_ids.to(device=features.device, dtype=torch.long)
        if view_ids.numel() > 0:
            min_id = int(view_ids.min().item())
            max_id = int(view_ids.max().item())
            if min_id < 0 or max_id >= self.num_view_slots:
                raise ValueError(
                    f"`view_ids` must be in [0,{self.num_view_slots - 1}], got min={min_id}, max={max_id}."
                )
        emb = self.view_embedding(view_ids).to(dtype=features.dtype)
        return features + emb.view(1, 1, int(features.shape[2]), int(features.shape[3]))

    def encode(
        self,
        features: torch.Tensor,
        view_ids: torch.Tensor,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        return self.adapter.encode(self._apply_view_embedding(features, view_ids))

    def decode(self, z_l: torch.Tensor) -> torch.Tensor:
        return self.adapter.decode(z_l)

    def forward(
        self,
        features: torch.Tensor,
        view_ids: torch.Tensor,
        *,
        return_reconstruction: bool = False,
    ):
        enc_out = self.encode(features, view_ids)
        if not return_reconstruction:
            return enc_out
        z_l = enc_out[0] if isinstance(enc_out, tuple) else enc_out
        return enc_out, self.decode(z_l)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def resolve_checkpoint_path(path: str | os.PathLike[str]) -> Path:
    ckpt_path = Path(os.path.expandvars(str(path))).expanduser()
    if not ckpt_path.is_absolute():
        ckpt_path = _repo_root() / ckpt_path
    return ckpt_path


def load_adapter_checkpoint_payload(path: str | os.PathLike[str]) -> dict[str, Any]:
    ckpt_path = resolve_checkpoint_path(path)
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"`jepa_adapter.checkpoint_path` does not exist: {ckpt_path}")
    payload = torch.load(str(ckpt_path), map_location="cpu")
    if not isinstance(payload, dict):
        raise ValueError(f"Invalid adapter checkpoint payload type from {ckpt_path}: {type(payload)}")
    return payload


def adapter_config_from_checkpoint(path: str | os.PathLike[str]) -> dict[str, Any]:
    payload = load_adapter_checkpoint_payload(path)
    config = payload.get("adapter_config")
    if not isinstance(config, dict):
        raise ValueError(f"`adapter_config` must be a dict in adapter checkpoint: {path}")
    return dict(config)


def adapter_latent_dim_from_checkpoint(path: str | os.PathLike[str]) -> int:
    config = adapter_config_from_checkpoint(path)
    if "adapter_latent_dim" not in config:
        raise ValueError(f"`adapter_config.adapter_latent_dim` missing in adapter checkpoint: {path}")
    return int(config["adapter_latent_dim"])


def _strip_checkpoint_prefix(key: str) -> str:
    prefixes = ("module.", "_orig_mod.", "adapter.", "model.")
    changed = True
    while changed:
        changed = False
        for prefix in prefixes:
            if key.startswith(prefix):
                key = key[len(prefix):]
                changed = True
    return key


class FrozenJepaAdapter(nn.Module):
    def __init__(
        self,
        adapter: SVAEAdapter,
        checkpoint_path: Path,
        freeze: bool = True,
        view_embedding: Optional[nn.Embedding] = None,
        view_schema: Optional[Sequence[str]] = None,
    ) -> None:
        super().__init__()
        self.adapter = adapter
        self.view_embedding = view_embedding
        self.view_schema = list(view_schema or [])
        self.checkpoint_path = str(checkpoint_path)
        self.freeze = bool(freeze)
        self.input_dim = int(adapter.input_dim)
        self.latent_dim = int(adapter.latent_dim)
        self.has_view_embedding = self.view_embedding is not None
        self._warned_view_embedding_fallback = False
        if self.freeze:
            self.requires_grad_(False)
            super().train(False)

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str | os.PathLike[str],
        *,
        input_dim: int,
        device: str | torch.device,
        torch_dtype: torch.dtype,
        freeze: bool = True,
    ) -> "FrozenJepaAdapter":
        ckpt_path = resolve_checkpoint_path(checkpoint_path)
        payload = load_adapter_checkpoint_payload(ckpt_path)
        config = payload.get("adapter_config")
        if not isinstance(config, dict):
            raise ValueError(f"`adapter_config` must be a dict in adapter checkpoint: {ckpt_path}")
        adapter_type = str(config.get("adapter_type", "identity")).lower()
        if adapter_type != "svae":
            raise ValueError(f"LeapBotWA JEPA adapter currently supports semantic-wm `svae`, got {adapter_type}.")
        adapter = SVAEAdapter(
            input_dim=int(input_dim),
            latent_dim=int(config.get("adapter_latent_dim", 96)),
            num_heads=int(config.get("adapter_num_heads", 12)),
            num_layers=int(config.get("adapter_num_layers", 3)),
            intermediate_size=int(config.get("adapter_intermediate_size", 3072)),
            progressive=bool(config.get("adapter_progressive", False)),
            mid_dim=config.get("adapter_mid_dim"),
            mid_heads=config.get("adapter_mid_heads"),
            latent_layers=int(config.get("adapter_latent_layers", 0)),
            latent_heads=config.get("adapter_latent_heads"),
        )
        state = payload.get("adapter") or payload.get("adapter_state_dict")
        if not isinstance(state, dict):
            raise ValueError(f"Adapter checkpoint missing `adapter` state dict: {ckpt_path}")
        stripped_state = {_strip_checkpoint_prefix(k): v for k, v in state.items()}
        missing, unexpected = adapter.load_state_dict(stripped_state, strict=False)
        if missing or unexpected:
            raise ValueError(
                f"Adapter state mismatch in {ckpt_path}: missing={missing[:10]}, unexpected={unexpected[:10]}"
            )
        view_embedding = None
        view_schema = payload.get("view_schema")
        if payload.get("view_embedding") is not None:
            logger.info(
                "Ignoring adapter checkpoint view_embedding; view identity is injected by JEPAMoTStateExpert."
            )

        wrapper = cls(
            adapter=adapter,
            checkpoint_path=ckpt_path,
            freeze=freeze,
            view_embedding=view_embedding,
            view_schema=view_schema if isinstance(view_schema, Sequence) and not isinstance(view_schema, (str, bytes)) else None,
        )
        wrapper.to(device=device, dtype=torch_dtype)
        wrapper.eval()
        logger.info(
            "Loaded frozen JEPA adapter: path=%s input_dim=%d latent_dim=%d freeze=%s",
            ckpt_path,
            wrapper.input_dim,
            wrapper.latent_dim,
            wrapper.freeze,
        )
        return wrapper

    def train(self, mode: bool = True):
        if self.freeze:
            super().train(False)
            self.adapter.eval()
            if self.view_embedding is not None:
                self.view_embedding.eval()
            return self
        return super().train(mode)

    def _apply_view_embedding(self, features: torch.Tensor, view_ids: Optional[torch.Tensor]) -> torch.Tensor:
        if self.view_embedding is None:
            return features
        if view_ids is None:
            raise ValueError("This JEPA adapter checkpoint contains view embeddings; `view_ids` must be provided.")
        if features.ndim != 4:
            raise ValueError(f"View-aware JEPA adapter expects features [B,T,N,C], got {tuple(features.shape)}.")
        if view_ids.ndim != 1 or int(view_ids.shape[0]) != int(features.shape[2]):
            raise ValueError(
                f"`view_ids` must be 1D with length N={features.shape[2]}, got {tuple(view_ids.shape)}."
            )
        view_ids = view_ids.to(device=features.device, dtype=torch.long)
        if view_ids.numel() > 0:
            min_id = int(view_ids.min().item())
            max_id = int(view_ids.max().item())
            if min_id < 0:
                raise ValueError(f"`view_ids` must be non-negative, got min={min_id}.")
        else:
            max_id = -1

        weight = self.view_embedding.weight.to(device=features.device, dtype=features.dtype)
        num_slots = int(weight.shape[0])
        if max_id >= num_slots:
            if not self._warned_view_embedding_fallback:
                logger.warning(
                    "JEPA adapter checkpoint has %d view-embedding slots but received view id up to %d. "
                    "Reusing view_id=1 embedding for unseen view ids so cross-dataset "
                    "per-view training can proceed. Consider training a Robotwin adapter for best quality.",
                    num_slots,
                    max_id,
                )
                self._warned_view_embedding_fallback = True
            fallback_index = 1 if num_slots > 1 else 0
            extra = weight[fallback_index:fallback_index + 1].expand(max_id + 1 - num_slots, -1)
            weight = torch.cat([weight, extra], dim=0)
        emb = F.embedding(view_ids, weight)
        return features + emb.view(1, 1, int(features.shape[2]), int(features.shape[3]))

    @torch.no_grad()
    def encode(self, features: torch.Tensor, view_ids: Optional[torch.Tensor] = None) -> torch.Tensor:
        if features.shape[-1] != self.input_dim:
            raise ValueError(
                f"JEPA adapter expected last dim {self.input_dim}, got {features.shape[-1]}."
            )
        out = self.adapter.encode(self._apply_view_embedding(features, view_ids))
        if isinstance(out, tuple):
            out = out[0]
        if out.shape[-1] != self.latent_dim:
            raise ValueError(
                f"JEPA adapter expected latent dim {self.latent_dim}, got {out.shape[-1]}."
            )
        return out

    def forward(self, features: torch.Tensor, view_ids: Optional[torch.Tensor] = None) -> torch.Tensor:
        return self.encode(features, view_ids=view_ids)
