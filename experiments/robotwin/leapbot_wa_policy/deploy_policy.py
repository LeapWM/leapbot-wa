import hashlib
import logging
import os
import sys
import time
import inspect
from collections import deque
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[3]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from leapbot_wa.datasets.lerobot.processors.leapbot_wa_processor import LeapBotWAProcessor
from experiments import DEFAULT_PROMPT
from leapbot_wa.datasets.lerobot.utils.normalizer import load_dataset_stats_from_json

logger = logging.getLogger(__name__)


def _is_none_like(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return value.strip().lower() in {"", "none", "null"}
    return False


def _parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"1", "true", "yes", "y"}:
            return True
        if lowered in {"0", "false", "no", "n"}:
            return False
    raise ValueError(f"Cannot parse bool value: {value}")


def _parse_optional_int(value: Any) -> Optional[int]:
    if _is_none_like(value):
        return None
    return int(value)


def _parse_optional_float(value: Any) -> Optional[float]:
    if _is_none_like(value):
        return None
    return float(value)


def _normalize_mixed_precision(mixed_precision: str) -> str:
    key = str(mixed_precision).strip().lower()
    if key not in {"no", "fp16", "bf16"}:
        raise ValueError(
            f"Unsupported mixed_precision: {mixed_precision}. "
            "Expected one of: ['no', 'fp16', 'bf16']."
        )
    return key


def _mixed_precision_to_model_dtype(mixed_precision: str) -> torch.dtype:
    precision = _normalize_mixed_precision(mixed_precision)
    if precision == "no":
        return torch.float32
    if precision == "fp16":
        return torch.float16
    return torch.bfloat16


def _resolve_sim_cfg_name(sim_cfg_path: Optional[str], sim_cfg_name: Optional[str]) -> str:
    configs_root = (PROJECT_ROOT / "configs").resolve()
    if not _is_none_like(sim_cfg_path):
        cfg_path = Path(str(sim_cfg_path)).expanduser().resolve()
        try:
            relative = cfg_path.relative_to(configs_root)
        except ValueError:
            return str(cfg_path)
        return relative.as_posix()

    if _is_none_like(sim_cfg_name):
        return "sim_robotwin.yaml"
    return str(sim_cfg_name)


def _compose_sim_cfg(
    sim_cfg_path: Optional[str],
    sim_cfg_name: Optional[str],
    sim_task: Optional[str],
) -> DictConfig:
    config_name = _resolve_sim_cfg_name(sim_cfg_path=sim_cfg_path, sim_cfg_name=sim_cfg_name)
    if Path(config_name).is_absolute():
        cfg = OmegaConf.load(config_name)
        OmegaConf.resolve(cfg)
        return cfg

    configs_root = (PROJECT_ROOT / "configs").resolve()
    overrides = []
    if not _is_none_like(sim_task):
        overrides.append(f"task={str(sim_task)}")

    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()

    with initialize_config_dir(version_base="1.3", config_dir=str(configs_root)):
        cfg = compose(config_name=config_name, overrides=overrides)
    return cfg


def _resolve_dataset_stats_path(dataset_stats_path: Optional[str]) -> Path:
    if _is_none_like(dataset_stats_path):
        raise FileNotFoundError(
            "`dataset_stats_path` is required. "
            "Please pass it from eval entrypoint overrides."
        )
    resolved = Path(str(dataset_stats_path)).expanduser().resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"Dataset stats path not found: {resolved}")
    return resolved


def _resize_rgb(image: np.ndarray, size_wh: tuple[int, int]) -> np.ndarray:
    pil_image = Image.fromarray(image.astype(np.uint8), mode="RGB")
    resized = pil_image.resize(size_wh, resample=Image.BICUBIC)
    return np.asarray(resized, dtype=np.uint8)


class LeapBotWARoboTwinPolicy:
    def __init__(
        self,
        model_cfg: DictConfig,
        processor_cfg: DictConfig,
        checkpoint_path: str,
        dataset_stats_path: Path,
        device: str,
        model_dtype: torch.dtype,
        action_horizon: int,
        replan_steps: int,
        num_inference_steps: int,
        sigma_shift: Optional[float],
        seed: Optional[int],
        text_cfg_scale: float,
        negative_prompt: str,
        rand_device: str,
        tiled: bool,
        timing_enabled: bool,
        num_video_frames: int,
        camera_view_keys: tuple[str, ...],
    ) -> None:
        model_cfg_copy = OmegaConf.create(OmegaConf.to_container(model_cfg, resolve=True))
        model_cfg_copy.load_text_encoder = True

        self.model = instantiate(model_cfg_copy, model_dtype=model_dtype, device=device)
        self.model.load_checkpoint(checkpoint_path)
        self.model = self.model.to(device).eval()

        self.processor: LeapBotWAProcessor = instantiate(processor_cfg).eval()
        dataset_stats = load_dataset_stats_from_json(str(dataset_stats_path))
        self.processor.set_normalizer_from_stats(dataset_stats)

        self.action_horizon = int(action_horizon)
        self.replan_steps = int(max(1, min(replan_steps, action_horizon)))
        self.num_inference_steps = int(num_inference_steps)
        self.sigma_shift = sigma_shift
        self.seed = seed
        self.text_cfg_scale = float(text_cfg_scale)
        self.negative_prompt = str(negative_prompt)
        self.rand_device = str(rand_device)
        self.tiled = bool(tiled)
        self.timing_enabled = bool(timing_enabled)
        self._num_video_frames = int(num_video_frames)
        self.camera_view_keys = tuple(camera_view_keys)

        self.pending_actions: deque[np.ndarray] = deque()
        self.pending_action_context: deque[Dict[str, int]] = deque()
        self.episode_count = 0
        self.step_count = 0
        self.inference_count = 0
        self._last_inference_record: Dict[str, Any] = {}
        self._timing_rollout = {"infer_s": 0.0, "sim_s": 0.0}

        logger.info(
            "Initialized LeapBotWARoboTwinPolicy | ckpt=%s | stats=%s | horizon=%d | replan=%d",
            checkpoint_path,
            dataset_stats_path,
            self.action_horizon,
            self.replan_steps,
        )

    def _normalize_state(self, state: np.ndarray) -> torch.Tensor:
        state_meta = self.processor.shape_meta["state"]
        if len(state_meta) != 1:
            raise ValueError("Expected exactly one merged state key in shape_meta['state'].")
        state_key = state_meta[0]["key"]

        state_batch = {"state": {state_key: torch.as_tensor(state, dtype=torch.float32).unsqueeze(0)}}
        state_batch = self.processor.action_state_transform(state_batch)
        state_batch = self.processor.normalizer.forward(state_batch)
        return state_batch["state"][state_key]

    def _denormalize_action(self, action: torch.Tensor) -> np.ndarray:
        if action.ndim == 2:
            action = action.unsqueeze(0)
        if action.ndim != 3:
            raise ValueError(f"Expected action tensor [B,T,D], got {tuple(action.shape)}")

        action_meta = self.processor.shape_meta["action"]
        if len(action_meta) != 1:
            raise ValueError("Expected exactly one merged action key in shape_meta['action'].")

        action_key = action_meta[0]["key"]
        normalizer = self.processor.normalizer.normalizers["action"][action_key]
        denorm = normalizer.backward(action.to(dtype=torch.float32, device="cpu"))
        return denorm.numpy()

    def _build_robotwin_per_view_jepa_tensor(self, observation: Dict[str, Any]) -> torch.Tensor:
        obs_data = observation["observation"]
        obs_key_by_train_key = {
            "cam_high": "head_camera",
            "head_camera": "head_camera",
            "cam_left_wrist": "left_camera",
            "left_camera": "left_camera",
            "cam_right_wrist": "right_camera",
            "right_camera": "right_camera",
        }
        view_tensors = []
        for train_key in self.camera_view_keys:
            obs_key = obs_key_by_train_key.get(train_key)
            if obs_key is None:
                raise KeyError(
                    f"Unsupported RoboTwin camera key `{train_key}` from training config. "
                    f"Known keys: {sorted(obs_key_by_train_key)}"
                )
            image = _resize_rgb(obs_data[obs_key]["rgb"], (320, 240))
            view_tensors.append(torch.from_numpy(image).permute(2, 0, 1))
        image_tensor = torch.stack(view_tensors, dim=0).unsqueeze(0).to(
            device=self.model.device,
            dtype=self.model.torch_dtype,
        )
        image_tensor = image_tensor * (2.0 / 255.0) - 1.0
        return image_tensor

    def _infer_action_chunk(
        self,
        observation: Dict[str, Any],
        instruction: str,
        collect_diagnostics: bool = False,
    ) -> np.ndarray:
        jepa_input_image = self._build_robotwin_per_view_jepa_tensor(observation)
        image_tensor = jepa_input_image[:, 0]
        state_vector = np.asarray(observation["joint_action"]["vector"], dtype=np.float32)
        proprio = self._normalize_state(state_vector)

        prompt = DEFAULT_PROMPT.format(task=instruction)
        infer_kwargs = {
            "prompt": prompt,
            "input_image": image_tensor,
            "action_horizon": self.action_horizon,
            "proprio": proprio,
            "negative_prompt": self.negative_prompt,
            "text_cfg_scale": self.text_cfg_scale,
            "num_inference_steps": self.num_inference_steps,
            "sigma_shift": self.sigma_shift,
            "seed": self.seed,
            "rand_device": self.rand_device,
            "tiled": self.tiled,
        }
        infer_action_params = inspect.signature(self.model.infer_action).parameters
        if "jepa_input_image" in infer_action_params:
            infer_kwargs["jepa_input_image"] = jepa_input_image
        if "num_video_frames" in infer_action_params:
            infer_kwargs["num_video_frames"] = int(self._num_video_frames)
        infer_t0 = time.perf_counter() if self.timing_enabled else 0.0
        with torch.no_grad():
            pred = self.model.infer_action(**infer_kwargs)
        if self.timing_enabled:
            self._timing_rollout["infer_s"] += time.perf_counter() - infer_t0

        action_tensor = pred["action"]  # [T, D]
        action_chunk = self._denormalize_action(action_tensor)[0]  # [T, D]
        inference_id = int(self.inference_count)
        self.inference_count += 1
        self._last_inference_record = {"inference_id": inference_id}
        if collect_diagnostics:
            normalized_action = action_tensor.detach().to(dtype=torch.float32, device="cpu").numpy()
            state_aligned = state_vector[:action_chunk.shape[1]]
            previous_and_chunk = np.vstack((state_aligned[None], action_chunk))
            self._last_inference_record.update({
                "instruction": str(instruction),
                "state_raw": state_vector,
                "proprio_normalized": proprio.detach().to(dtype=torch.float32, device="cpu").numpy(),
                "action_normalized": normalized_action,
                "action_denormalized": action_chunk,
                "target_delta_from_observation": action_chunk - state_aligned[None],
                "consecutive_action_delta": np.diff(previous_and_chunk, axis=0),
                "all_finite_normalized": bool(np.all(np.isfinite(normalized_action))),
                "all_finite_denormalized": bool(np.all(np.isfinite(action_chunk))),
            })
        return action_chunk

    def _fill_action_queue(
        self, task_env, observation: Dict[str, Any], instruction: str
    ) -> None:
        collect_diagnostics = bool(getattr(task_env, "diagnostic_log_enabled", False))
        action_chunk = self._infer_action_chunk(
            observation=observation,
            instruction=instruction,
            collect_diagnostics=collect_diagnostics,
        )
        n_exec = min(self.replan_steps, action_chunk.shape[0])
        if collect_diagnostics:
            observation_images = {}
            observation_camera_config = {}
            for camera_name, camera_data in observation["observation"].items():
                rgb = camera_data.get("rgb")
                if rgb is not None:
                    rgb_array = np.asarray(rgb)
                    observation_images[camera_name] = {
                        "shape": list(rgb_array.shape),
                        "dtype": str(rgb_array.dtype),
                        "min": float(np.min(rgb_array)),
                        "max": float(np.max(rgb_array)),
                        "mean": float(np.mean(rgb_array)),
                        "std": float(np.std(rgb_array)),
                        "sha256": hashlib.sha256(rgb_array.tobytes()).hexdigest(),
                    }
                observation_camera_config[camera_name] = {
                    key: value for key, value in camera_data.items() if key != "rgb"
                }
            task_env.log_diagnostic(
                "model_inference",
                **self._last_inference_record,
                action_horizon=int(action_chunk.shape[0]),
                executed_chunk_length=int(n_exec),
                observation_images=observation_images,
                observation_camera_config=observation_camera_config,
                simulator_snapshot=task_env._safe_diagnostic_snapshot(),
            )
        inference_id = int(self._last_inference_record["inference_id"])
        for i in range(n_exec):
            self.pending_actions.append(np.asarray(action_chunk[i], dtype=np.float32))
            self.pending_action_context.append(
                {"inference_id": inference_id, "chunk_index": int(i)}
            )

    def should_request_observation(self) -> bool:
        return not self.pending_actions

    def step(self, task_env, observation: Optional[Dict[str, Any]]) -> None:
        if not self.pending_actions:
            if observation is None:
                raise ValueError(
                    "Observation is required when action queue is empty "
                    "(replan step for leapbot_wa)."
                )
            instruction = task_env.get_instruction()
            self._fill_action_queue(
                task_env=task_env, observation=observation, instruction=instruction
            )

        if not self.pending_actions:
            logger.warning("No action generated; skip current eval step.")
            return

        action = self.pending_actions.popleft()
        action_context = self.pending_action_context.popleft()
        log_diagnostic = getattr(task_env, "log_diagnostic", lambda *args, **kwargs: None)
        log_diagnostic(
            "policy_action",
            model_step=int(self.step_count),
            pending_actions_after_pop=int(len(self.pending_actions)),
            target_action=action,
            **action_context,
        )
        sim_t0 = time.perf_counter() if self.timing_enabled else 0.0
        task_env.take_action(action, action_type="qpos")
        if self.timing_enabled:
            self._timing_rollout["sim_s"] += time.perf_counter() - sim_t0
        self.step_count += 1

    def reset_timing_rollout(self) -> None:
        self._timing_rollout["infer_s"] = 0.0
        self._timing_rollout["sim_s"] = 0.0

    def get_timing_rollout(self) -> Dict[str, float]:
        return {
            "infer_s": float(self._timing_rollout["infer_s"]),
            "sim_s": float(self._timing_rollout["sim_s"]),
        }

    def reset(self) -> None:
        self.pending_actions.clear()
        self.pending_action_context.clear()
        self.episode_count += 1
        self.step_count = 0
        self.inference_count = 0
        self._last_inference_record = {}
        self.reset_timing_rollout()


def encode_obs(observation: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    return observation


def get_model(usr_args: Dict[str, Any]):
    sim_cfg_path = usr_args.get("sim_cfg_path")
    sim_cfg_name = usr_args.get("sim_cfg_name")
    sim_task = usr_args.get("sim_task")
    cfg = _compose_sim_cfg(
        sim_cfg_path=sim_cfg_path,
        sim_cfg_name=sim_cfg_name,
        sim_task=sim_task,
    )

    checkpoint_path = usr_args.get("ckpt_setting")
    if _is_none_like(checkpoint_path):
        raise ValueError("`ckpt_setting` is required and must be a valid checkpoint path.")

    device = str(usr_args.get("device") or cfg.EVALUATION.get("device") or "cuda")
    if device.startswith("cuda") and not torch.cuda.is_available():
        logger.warning("CUDA is unavailable; fallback device to cpu.")
        device = "cpu"

    mixed_precision = str(usr_args.get("mixed_precision") or cfg.get("mixed_precision", "bf16"))
    model_dtype = _mixed_precision_to_model_dtype(mixed_precision)

    dataset_stats_path = _resolve_dataset_stats_path(
        dataset_stats_path=usr_args.get("dataset_stats_path"),
    )

    action_horizon = _parse_optional_int(usr_args.get("action_horizon"))
    if action_horizon is None:
        eval_horizon = _parse_optional_int(cfg.EVALUATION.get("action_horizon"))
        action_horizon = eval_horizon if eval_horizon is not None else (int(cfg.processor.num_obs_steps) - 1)
    if action_horizon <= 0:
        raise ValueError(f"`action_horizon` must be positive, got {action_horizon}")

    replan_steps = _parse_optional_int(usr_args.get("replan_steps"))
    if replan_steps is None:
        replan_steps = int(cfg.EVALUATION.get("replan_steps", 10))

    num_inference_steps = _parse_optional_int(usr_args.get("num_inference_steps"))
    if num_inference_steps is None:
        num_inference_steps = int(cfg.EVALUATION.get("num_inference_steps", 10))

    sigma_shift = _parse_optional_float(usr_args.get("sigma_shift"))
    if sigma_shift is None:
        sigma_shift = _parse_optional_float(cfg.EVALUATION.get("sigma_shift"))

    seed = _parse_optional_int(usr_args.get("seed"))
    text_cfg_scale = float(usr_args.get("text_cfg_scale", cfg.EVALUATION.get("text_cfg_scale", 1.0)))
    negative_prompt = str(usr_args.get("negative_prompt", cfg.EVALUATION.get("negative_prompt", "")))
    rand_device = str(usr_args.get("rand_device", cfg.EVALUATION.get("rand_device", "cpu")))
    tiled = _parse_bool(usr_args.get("tiled", cfg.EVALUATION.get("tiled", False)))
    timing_enabled = _parse_bool(
        usr_args.get("timing_enabled", cfg.EVALUATION.get("timing_enabled", False))
    )

    train_image_meta = OmegaConf.to_container(cfg.processor.shape_meta.images, resolve=True)
    camera_view_keys = tuple(str(meta["key"]) for meta in train_image_meta)

    policy = LeapBotWARoboTwinPolicy(
        model_cfg=cfg.model,
        processor_cfg=cfg.processor,
        checkpoint_path=str(checkpoint_path),
        dataset_stats_path=dataset_stats_path,
        device=device,
        model_dtype=model_dtype,
        action_horizon=action_horizon,
        replan_steps=replan_steps,
        num_inference_steps=num_inference_steps,
        sigma_shift=sigma_shift,
        seed=seed,
        text_cfg_scale=text_cfg_scale,
        negative_prompt=negative_prompt,
        rand_device=rand_device,
        tiled=tiled,
        timing_enabled=timing_enabled,
        num_video_frames=((int(cfg.processor.num_obs_steps) - 1)) // 4 + 1,
        camera_view_keys=camera_view_keys,
    )
    return policy


def eval(TASK_ENV, model, observation: Optional[Dict[str, Any]]):
    obs = encode_obs(observation)
    model.step(TASK_ENV, obs)


def reset_model(model):
    model.reset()
