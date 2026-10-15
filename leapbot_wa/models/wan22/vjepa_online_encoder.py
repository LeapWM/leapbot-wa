from __future__ import annotations

import sys
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Optional

import torch
import torch.nn as nn

from leapbot_wa.utils.logging_config import get_logger
from .jepa_adapter import resolve_checkpoint_path

logger = get_logger(__name__)


DEFAULT_VJEPA2_ROOT = os.environ.get("LEAPBOT_WA_VJEPA_ROOT", str(Path(__file__).resolve().parents[2] / "_vendor" / "vjepa2"))
DEFAULT_VJEPA2_CHECKPOINT = os.environ.get("LEAPBOT_WA_VJEPA_CHECKPOINT", "weights/vjepa_latest.pt")
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def _clean_vjepa_state_dict(state_dict: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {
        str(key).replace("module.", "").replace("backbone.", ""): value
        for key, value in state_dict.items()
    }


def _as_torch_dtype(value: Any, default: torch.dtype) -> torch.dtype:
    if isinstance(value, torch.dtype):
        return value
    if value is None or str(value).strip().lower() in {"", "auto", "model"}:
        return default
    mapping = {
        "fp32": torch.float32,
        "float32": torch.float32,
        "fp16": torch.float16,
        "float16": torch.float16,
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
    }
    key = str(value).strip().lower()
    if key not in mapping:
        raise ValueError(f"Unsupported V-JEPA dtype: {value!r}")
    return mapping[key]


class VJEPAOnlineEncoderWrapper(nn.Module):
    """Frozen V-JEPA2 encoder for online current/future full patch tokens.

    The V-JEPA encoder is intentionally kept out of this module's registered
    submodules so `model.state_dict()` and optimizer collection do not include
    the frozen teacher weights.
    """

    def __init__(
        self,
        *,
        vjepa2_root: str = DEFAULT_VJEPA2_ROOT,
        checkpoint_path: str = DEFAULT_VJEPA2_CHECKPOINT,
        image_size: int = 384,
        patch_size: int = 16,
        tubelet_size: int = 2,
        vjepa_dim: int = 1408,
        num_jepa_tokens: Optional[int] = None,
        dtype: torch.dtype | str = torch.bfloat16,
        device: str | torch.device = "cuda",
    ):
        super().__init__()
        self.vjepa2_root = str(vjepa2_root)
        self.checkpoint_path = str(checkpoint_path)
        self.image_size = int(image_size)
        self.patch_size = int(patch_size)
        self.tubelet_size = int(tubelet_size)
        self.vjepa_dim = int(vjepa_dim)
        self.dtype = _as_torch_dtype(dtype, torch.bfloat16)
        self.device_name = str(device)

        if self.image_size <= 0 or self.patch_size <= 0 or self.tubelet_size <= 0:
            raise ValueError("V-JEPA image_size, patch_size, and tubelet_size must be positive.")
        if self.image_size % self.patch_size != 0:
            raise ValueError(
                f"V-JEPA image_size must be divisible by patch_size, got {self.image_size} and {self.patch_size}."
            )
        self.reference_grid_size = (self.image_size // self.patch_size, self.image_size // self.patch_size)
        inferred_tokens = self.reference_grid_size[0] * self.reference_grid_size[1]
        self.num_jepa_tokens = int(num_jepa_tokens) if num_jepa_tokens is not None else int(inferred_tokens)
        if self.num_jepa_tokens != inferred_tokens:
            raise ValueError(
                "Configured `num_jepa_tokens` must match the reference V-JEPA spatial tokens: "
                f"configured={self.num_jepa_tokens}, inferred={inferred_tokens}."
            )

        root = Path(self.vjepa2_root).expanduser().resolve()
        checkpoint = resolve_checkpoint_path(self.checkpoint_path).resolve()
        if not root.exists():
            raise FileNotFoundError(f"V-JEPA2 root does not exist: {root}")
        if not checkpoint.is_file():
            raise FileNotFoundError(f"V-JEPA2 checkpoint does not exist: {checkpoint}")

        payload = torch.load(str(checkpoint), map_location="cpu")
        encoder_key = "target_encoder" if "target_encoder" in payload else "encoder"
        if encoder_key not in payload:
            raise KeyError(f"Checkpoint {checkpoint} does not contain `target_encoder` or `encoder`.")
        encoder_state_dict = _clean_vjepa_state_dict(payload[encoder_key])
        checkpoint_dim = self._infer_checkpoint_dim(encoder_state_dict)
        if checkpoint_dim is not None and int(checkpoint_dim) != self.vjepa_dim:
            raise ValueError(
                f"V-JEPA checkpoint dim mismatch: config={self.vjepa_dim}, checkpoint={checkpoint_dim}."
            )

        is_vjepa2_1 = self._is_vjepa2_1_state_dict(encoder_state_dict)
        if is_vjepa2_1 and self.dtype != torch.float32:
            logger.warning(
                "Forcing V-JEPA2.1 encoder dtype from %s to torch.float32 because its RoPE/SDPA path "
                "can produce mixed q/k/v dtypes in bf16/fp16. Downstream JEPA tokens are still cast to model dtype.",
                self.dtype,
            )
            self.dtype = torch.float32

        encoder = self._build_encoder(root=root, state_dict=encoder_state_dict)
        strict_load = is_vjepa2_1
        missing, unexpected = encoder.load_state_dict(
            encoder_state_dict,
            strict=strict_load,
        )
        logger.info(
            "Loaded frozen V-JEPA2 encoder from %s key `%s` strict=%s: missing=%d unexpected=%d",
            checkpoint,
            encoder_key,
            strict_load,
            len(missing),
            len(unexpected),
        )

        actual_patch_size = int(getattr(encoder, "patch_size", self.patch_size))
        actual_tubelet_size = int(getattr(encoder, "tubelet_size", self.tubelet_size))
        if actual_patch_size != self.patch_size:
            raise ValueError(f"V-JEPA patch_size mismatch: config={self.patch_size}, encoder={actual_patch_size}")
        if actual_tubelet_size != self.tubelet_size:
            raise ValueError(
                f"V-JEPA tubelet_size mismatch: config={self.tubelet_size}, encoder={actual_tubelet_size}"
            )

        encoder.to(device=self.device_name, dtype=self.dtype).eval()
        encoder.requires_grad_(False)
        self.__dict__["_encoder"] = encoder

        self.register_buffer(
            "image_mean",
            torch.tensor(IMAGENET_MEAN, dtype=torch.float32).view(1, 3, 1, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "image_std",
            torch.tensor(IMAGENET_STD, dtype=torch.float32).view(1, 3, 1, 1, 1),
            persistent=False,
        )
        self.train(False)

    @property
    def encoder(self) -> nn.Module:
        return self.__dict__["_encoder"]

    @staticmethod
    def _is_vjepa2_1_state_dict(state_dict: Mapping[str, torch.Tensor]) -> bool:
        return any(
            key in state_dict
            for key in ("img_mod_embed", "video_mod_embed", "patch_embed_img.proj.weight")
        )

    @staticmethod
    def _infer_checkpoint_dim(state_dict: Mapping[str, torch.Tensor]) -> Optional[int]:
        for key in (
            "patch_embed.proj.weight",
            "blocks.0.attn.qkv.weight",
            "norm.weight",
            "norms_block.0.weight",
        ):
            value = state_dict.get(key)
            if value is None:
                continue
            if key == "blocks.0.attn.qkv.weight":
                return int(value.shape[0] // 3)
            return int(value.shape[0])
        return None

    def _build_encoder(self, *, root: Path, state_dict: Mapping[str, torch.Tensor]) -> nn.Module:
        if not self._is_vjepa2_1_state_dict(state_dict) or self.vjepa_dim != 1664:
            raise ValueError("The released model requires the mixed V-JEPA2.1 1664-dim encoder")
        from leapbot_wa._vendor.vjepa2.app.vjepa_2_1.models import vision_transformer as vit_encoder
        return vit_encoder.vit_gigantic_xformers(
            patch_size=self.patch_size,
            img_size=(self.image_size, self.image_size),
            num_frames=64,
            tubelet_size=self.tubelet_size,
            use_sdpa=True,
            uniform_power=False,
            use_rope=True,
            img_temporal_dim_size=1,
            interpolate_rope=True,
            modality_embedding=True,
        )

    def train(self, mode: bool = True):
        super().train(False)
        if "_encoder" in self.__dict__:
            self.encoder.eval()
        return self

    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)
        if "_encoder" in self.__dict__:
            self.encoder.to(*args, **kwargs)
        return self

    def _grid_size_from_clip(self, clip: torch.Tensor) -> tuple[int, int]:
        height, width = int(clip.shape[-2]), int(clip.shape[-1])
        if height % self.patch_size != 0 or width % self.patch_size != 0:
            raise ValueError(
                "V-JEPA input spatial dims must be divisible by patch_size: "
                f"got HxW=({height},{width}), patch_size={self.patch_size}."
            )
        return height // self.patch_size, width // self.patch_size

    def _preprocess_clip(self, clip: torch.Tensor) -> torch.Tensor:
        if clip.ndim != 5:
            raise ValueError(f"V-JEPA clip must be [B,3,T,H,W], got {tuple(clip.shape)}")
        if clip.shape[1] != 3:
            raise ValueError(f"V-JEPA clip must have 3 channels, got {clip.shape[1]}")
        clip = clip.to(device=self.device_name, dtype=torch.float32, non_blocking=True)
        clip = ((clip + 1.0) * 0.5).clamp(0.0, 1.0)
        mean = self.image_mean.to(device=clip.device, dtype=clip.dtype)
        std = self.image_std.to(device=clip.device, dtype=clip.dtype)
        clip = (clip - mean) / std
        return clip.to(dtype=self.dtype)

    @torch.no_grad()
    def encode_clip(self, clip: torch.Tensor, *, return_grid: bool = False):
        grid_size = self._grid_size_from_clip(clip)
        frames = int(clip.shape[2])
        if frames % self.tubelet_size != 0:
            raise ValueError(
                "V-JEPA clip length must be divisible by tubelet_size: "
                f"frames={frames}, tubelet_size={self.tubelet_size}."
            )
        clip = self._preprocess_clip(clip)
        tokens = self.encoder(clip)
        if isinstance(tokens, (list, tuple)):
            if len(tokens) != 1:
                raise ValueError(f"Expected one V-JEPA encoder output, got {len(tokens)}")
            tokens = tokens[0]
        if tokens.ndim != 3:
            raise ValueError(f"V-JEPA encoder output must be [B,S,D], got {tuple(tokens.shape)}")
        if tokens.shape[-1] != self.vjepa_dim:
            raise ValueError(f"V-JEPA dim mismatch: expected {self.vjepa_dim}, got {tokens.shape[-1]}")
        expected_seq = (frames // self.tubelet_size) * grid_size[0] * grid_size[1]
        if int(tokens.shape[1]) != int(expected_seq):
            raise ValueError(
                "V-JEPA token count mismatch for dynamic grid: "
                f"expected={expected_seq} from frames={frames}, grid={grid_size}, got={tokens.shape[1]}."
            )
        tokens = tokens.detach()
        if return_grid:
            return tokens, grid_size
        return tokens

    @torch.no_grad()
    def encode_current(self, video: torch.Tensor, *, return_grid: bool = False):
        if video.ndim == 4:
            video = video.unsqueeze(2)
        if video.ndim != 5:
            raise ValueError(f"`video` must be [B,3,T,H,W] or [B,3,H,W], got {tuple(video.shape)}")
        if video.shape[2] < 1:
            raise ValueError("`video` must contain at least one frame for current V-JEPA encoding.")
        current_clip = video[:, :, 0:1].repeat(1, 1, self.tubelet_size, 1, 1)
        current_tokens, grid_size = self.encode_clip(current_clip, return_grid=True)
        expected_tokens = grid_size[0] * grid_size[1]
        if current_tokens.shape[1] != expected_tokens:
            raise ValueError(
                f"Current V-JEPA token count mismatch: expected {expected_tokens}, got {current_tokens.shape[1]}."
            )
        if return_grid:
            return current_tokens, grid_size
        return current_tokens

    @torch.no_grad()
    def forward(
        self,
        video: torch.Tensor,
        current_frame_policy: str = "static_2frame_first",
        future_frame_policy: str = "future_clip_tubelet_aligned",
        num_future_steps: Optional[int] = None,
        return_future: bool = True,
        future_mask: Optional[torch.Tensor] = None,
    ) -> dict[str, torch.Tensor | None]:
        if current_frame_policy != "static_2frame_first":
            raise ValueError(f"Unsupported current_frame_policy: {current_frame_policy}")
        if future_frame_policy != "future_clip_tubelet_aligned":
            raise ValueError(f"Unsupported future_frame_policy: {future_frame_policy}")
        if video.ndim != 5:
            raise ValueError(f"`video` must be [B,3,T,H,W], got {tuple(video.shape)}")
        if video.shape[1] != 3:
            raise ValueError(f"`video` channel dimension must be 3, got {video.shape[1]}")

        batch_size = int(video.shape[0])
        current_tokens, grid_size = self.encode_current(video, return_grid=True) # [B, N, D]
        tokens_per_frame = int(grid_size[0] * grid_size[1])

        future_tokens = None
        if return_future:
            future_clip = video[:, :, 1:] #[B, 3, T-1, H, W]
            future_frames = int(future_clip.shape[2])
            if future_frames <= 0:
                raise ValueError("Future V-JEPA clip must contain at least one frame.")
            if future_frames % self.tubelet_size != 0:
                raise ValueError(
                    "Future V-JEPA clip length must be divisible by tubelet_size: "
                    f"frames={future_frames}, tubelet_size={self.tubelet_size}."
                )
            future_steps = future_frames // self.tubelet_size
            if num_future_steps is not None and future_steps != int(num_future_steps):
                raise ValueError(
                    f"Future V-JEPA steps mismatch: expected {int(num_future_steps)}, got {future_steps}."
                )
            flat_future, future_grid = self.encode_clip(future_clip, return_grid=True) #[B, N*(Tf), D]
            if future_grid != grid_size:
                raise ValueError(f"Current/future V-JEPA grids mismatch: current={grid_size}, future={future_grid}.")
            expected_future_seq = future_steps * tokens_per_frame
            if flat_future.shape[1] != expected_future_seq:
                raise ValueError(
                    f"Future V-JEPA token count mismatch: expected {expected_future_seq}, got {flat_future.shape[1]}."
                )
            future_tokens = flat_future.reshape(batch_size, future_steps, tokens_per_frame, self.vjepa_dim)

        return {
            "current_tokens": current_tokens,
            "future_tokens": future_tokens,
            "future_mask": future_mask,
            "grid_size": grid_size,
            "tokens_per_frame": tokens_per_frame,
        }

class PerViewVJEPAEncoder(nn.Module):
    """Encode each camera view independently with V-JEPA and concatenate tokens.

    Input is canonical [B, V, 3, T, H, W]. Dataset code that produces
    [B, V, T, 3, H, W] should convert at the dataset/model boundary before
    calling this encoder.
    """

    def __init__(self, online_encoder: VJEPAOnlineEncoderWrapper, num_view_slots: int = 2):
        super().__init__()
        self.online_encoder = online_encoder
        self.num_view_slots = int(num_view_slots)
        if self.num_view_slots <= 0:
            raise ValueError(f"`num_view_slots` must be positive, got {self.num_view_slots}.")

    def _validate_video(self, video: torch.Tensor) -> tuple[int, int, int, int, int, int]:
        if video.ndim != 6:
            raise ValueError(f"Per-view V-JEPA input must be [B,V,3,T,H,W], got {tuple(video.shape)}.")
        batch_size, num_views, channels, frames, height, width = video.shape
        if channels != 3:
            raise ValueError(
                "Per-view V-JEPA input must be canonical [B,V,3,T,H,W]; "
                f"got shape {tuple(video.shape)} with dim2={channels}."
            )
        if num_views > self.num_view_slots:
            raise ValueError(f"num_views={num_views} exceeds num_view_slots={self.num_view_slots}.")
        return int(batch_size), int(num_views), int(channels), int(frames), int(height), int(width)

    @staticmethod
    def _make_view_ids(num_views: int, tokens_per_view: int, device: torch.device) -> torch.Tensor:
        return torch.arange(num_views, device=device, dtype=torch.long).repeat_interleave(tokens_per_view)

    @torch.no_grad()
    def encode_clip(self, video: torch.Tensor) -> dict[str, torch.Tensor | tuple[int, int] | int]:
        batch_size, num_views, channels, frames, height, width = self._validate_video(video)
        if frames % self.online_encoder.tubelet_size != 0:
            raise ValueError(
                "Per-view V-JEPA clip length must be divisible by tubelet_size: "
                f"frames={frames}, tubelet_size={self.online_encoder.tubelet_size}."
            )
        flat_video = video.reshape(batch_size * num_views, channels, frames, height, width)
        flat_tokens, grid_size = self.online_encoder.encode_clip(flat_video, return_grid=True)
        tokens_per_view = int(grid_size[0] * grid_size[1])
        steps = frames // self.online_encoder.tubelet_size
        expected_seq = steps * tokens_per_view
        if int(flat_tokens.shape[1]) != expected_seq:
            raise ValueError(
                "Per-view V-JEPA token count mismatch: "
                f"expected={expected_seq}, got={flat_tokens.shape[1]}."
            )
        tokens = flat_tokens.reshape(
            batch_size, num_views, steps, tokens_per_view, self.online_encoder.vjepa_dim
        )
        tokens = tokens.permute(0, 2, 1, 3, 4).reshape(
            batch_size, steps, num_views * tokens_per_view, self.online_encoder.vjepa_dim
        )
        view_ids = self._make_view_ids(num_views, tokens_per_view, tokens.device)
        return {
            "tokens": tokens,
            "view_ids": view_ids,
            "grid_size": grid_size,
            "tokens_per_view": tokens_per_view,
            "tokens_per_frame": num_views * tokens_per_view,
            "num_views": num_views,
            "steps": steps,
        }

    @torch.no_grad()
    def encode_leapbot_wa_current_future(
        self,
        video: torch.Tensor,
        *,
        num_future_steps: Optional[int] = None,
    ) -> dict[str, torch.Tensor | tuple[int, int] | int]:
        """Encode per-view tokens using LeapBotWA's current/future JEPA flow.

        The first sampled frame is repeated to one tubelet for the current state.
        Remaining frames are encoded as a tubelet-aligned future clip. For the
        usual LIBERO setting this consumes 9 sampled frames and returns 5 JEPA
        feature frames: 1 current + 4 future.
        """
        batch_size, num_views, channels, frames, height, width = self._validate_video(video)
        if frames < 2:
            raise ValueError(f"LeapBotWA per-view JEPA flow needs at least 2 sampled frames, got {frames}.")
        tubelet = int(self.online_encoder.tubelet_size)
        future_frames = frames - 1
        if future_frames % tubelet != 0:
            raise ValueError(
                "LeapBotWA future clip length must be divisible by tubelet_size after removing the first frame: "
                f"future_frames={future_frames}, tubelet_size={tubelet}."
            )
        future_steps = future_frames // tubelet
        if num_future_steps is not None and future_steps != int(num_future_steps):
            raise ValueError(f"Future steps mismatch: expected {int(num_future_steps)}, got {future_steps}.")

        current_clip = video[:, :, :, 0:1].repeat(1, 1, 1, tubelet, 1, 1)
        flat_current = current_clip.reshape(batch_size * num_views, channels, tubelet, height, width)
        current_tokens, grid_size = self.online_encoder.encode_clip(flat_current, return_grid=True)
        tokens_per_view = int(grid_size[0] * grid_size[1])
        if int(current_tokens.shape[1]) != tokens_per_view:
            raise ValueError(
                f"Current per-view token count mismatch: expected {tokens_per_view}, got {current_tokens.shape[1]}."
            )
        current_tokens = current_tokens.reshape(
            batch_size, num_views, tokens_per_view, self.online_encoder.vjepa_dim
        ).reshape(batch_size, num_views * tokens_per_view, self.online_encoder.vjepa_dim)

        future_clip = video[:, :, :, 1:]
        flat_future_clip = future_clip.reshape(batch_size * num_views, channels, future_frames, height, width)
        flat_future, future_grid = self.online_encoder.encode_clip(flat_future_clip, return_grid=True)
        if future_grid != grid_size:
            raise ValueError(f"Current/future V-JEPA grids mismatch: current={grid_size}, future={future_grid}.")
        expected_future_seq = future_steps * tokens_per_view
        if int(flat_future.shape[1]) != expected_future_seq:
            raise ValueError(
                f"Future per-view token count mismatch: expected {expected_future_seq}, got {flat_future.shape[1]}."
            )
        future_tokens = flat_future.reshape(
            batch_size, num_views, future_steps, tokens_per_view, self.online_encoder.vjepa_dim
        )
        future_tokens = future_tokens.permute(0, 2, 1, 3, 4).reshape(
            batch_size, future_steps, num_views * tokens_per_view, self.online_encoder.vjepa_dim
        )

        tokens = torch.cat([current_tokens.unsqueeze(1), future_tokens], dim=1)
        view_ids = self._make_view_ids(num_views, tokens_per_view, tokens.device)
        return {
            "tokens": tokens,
            "current_tokens": current_tokens,
            "future_tokens": future_tokens,
            "view_ids": view_ids,
            "grid_size": grid_size,
            "tokens_per_view": tokens_per_view,
            "tokens_per_frame": num_views * tokens_per_view,
            "num_views": num_views,
            "future_steps": future_steps,
            "steps": future_steps + 1,
        }
