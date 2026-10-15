"""Single-task LIBERO closed-loop evaluation with LeapBot-WA."""
import json
import logging
import os
import sys
import time
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import hydra
import numpy as np
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from PIL import Image
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.libero.libero_utils import (
    LIBERO_ENV_RESOLUTION, get_libero_dummy_action, get_libero_env,
    get_libero_image, invert_gripper_action, quat2axisangle, save_rollout_video,
)
from experiments.libero.action_ensembler import ActionEnsembler
from leapbot_wa.datasets.lerobot.processors.leapbot_wa_processor import LeapBotWAProcessor
from leapbot_wa.datasets.lerobot.utils.normalizer import load_dataset_stats_from_json
from experiments import DEFAULT_PROMPT

@dataclass
class EvalRuntime:
    model: torch.nn.Module
    processor: LeapBotWAProcessor
    action_horizon: int
    input_w: int
    input_h: int
    model_device: str
    dataset_stats_path: Path

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)

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

def _resolve_eval_device(cfg: DictConfig) -> str:
    eval_device = cfg.EVALUATION.get("device")
    if eval_device is not None:
        return str(eval_device)
    return "cuda" if torch.cuda.is_available() else "cpu"

def _resolve_dataset_stats_path(cfg: DictConfig) -> Path:
    explicit = cfg.EVALUATION.get("dataset_stats_path")
    candidates: list[Path] = []

    if explicit is not None:
        candidates.append(Path(os.path.expanduser(os.path.expandvars(str(explicit)))))

    ckpt = Path(os.path.expanduser(os.path.expandvars(str(cfg.ckpt))))
    for parent in list(ckpt.parents)[:4]:
        candidates.append(parent / "dataset_stats.json")

    seen = set()
    for path in candidates:
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        if resolved.exists():
            return resolved

    msg = (
        "Failed to locate dataset_stats.json. Tried explicit "
        "EVALUATION.dataset_stats_path and checkpoint parent directories. "
        "Please pass EVALUATION.dataset_stats_path=/path/to/dataset_stats.json."
    )
    raise FileNotFoundError(msg)

def _center_crop_resize(image: np.ndarray, width: int, height: int) -> np.ndarray:
    pil_image = Image.fromarray(image)
    src_w, src_h = pil_image.size
    scale = max(width / src_w, height / src_h)
    resized = pil_image.resize((round(src_w * scale), round(src_h * scale)), resample=Image.BILINEAR)
    rw, rh = resized.size
    left = max((rw - width) // 2, 0)
    top = max((rh - height) // 2, 0)
    cropped = resized.crop((left, top, left + width, top + height))
    return np.asarray(cropped, dtype=np.uint8)

def _normalize_proprio(
    proprio: np.ndarray,
    processor: LeapBotWAProcessor,
) -> torch.Tensor:
    state_meta = processor.shape_meta["state"]
    if len(state_meta) != 1:
        raise ValueError(
            "LIBERO eval currently expects a single merged state key in shape_meta['state']."
        )
    state_key = state_meta[0]["key"]

    state_batch = {"state": {state_key: torch.as_tensor(proprio, dtype=torch.float32).unsqueeze(0)}}
    state_batch = processor.action_state_transform(state_batch)
    state_batch = processor.normalizer.forward(state_batch)
    return state_batch["state"][state_key]

def _shape_meta_hw(meta: dict, camera_idx: int) -> tuple[int, int]:
    shape = meta["shape"]
    if len(shape) != 3:
        raise ValueError(f"shape_meta.images[{camera_idx}].shape must be [C,H,W], got {shape}")
    return int(shape[1]), int(shape[2])

def _shape_meta_key(meta: dict, camera_idx: int) -> str:
    key = meta.get("key") if hasattr(meta, "get") else None
    if key is not None:
        return str(key)
    return "image" if camera_idx == 0 else f"view_{camera_idx}"

def _obs_to_per_view_jepa_input(
    obs: dict,
    processor: LeapBotWAProcessor,
    device: str,
    dtype: torch.dtype,
) -> torch.Tensor:
    imgs = get_libero_image(obs)
    image_meta = processor.shape_meta["images"]
    num_cameras = int(processor.num_output_cameras)
    if len(image_meta) < num_cameras:
        raise ValueError(
            f"shape_meta.images has {len(image_meta)} entries, "
            f"but num_output_cameras={num_cameras}."
        )

    views = []
    for camera_idx in range(num_cameras):
        meta = image_meta[camera_idx]
        key = _shape_meta_key(meta, camera_idx)
        if key not in imgs:
            fallback_key = "image" if camera_idx == 0 else "wrist_image"
            if fallback_key in imgs:
                key = fallback_key
            else:
                raise KeyError(
                    f"Observation image key {key!r} missing and fallback {fallback_key!r} not found. "
                    f"Available keys: {sorted(imgs.keys())}."
                )
        target_h, target_w = _shape_meta_hw(meta, camera_idx)
        views.append(_center_crop_resize(imgs[key], width=target_w, height=target_h))

    stacked = np.stack(views, axis=0)
    x = torch.tensor(stacked).permute(0, 3, 1, 2).unsqueeze(0).to(device=device, dtype=dtype)
    return x * (2.0 / 255.0) - 1.0

def _extract_sim_state(obs: dict) -> np.ndarray:
    """Build simulator state from current observation.

    This is used as proprio input for model inference.
    """
    state = np.concatenate(
        (
            obs["robot0_eef_pos"],
            quat2axisangle(obs["robot0_eef_quat"]),
            obs["robot0_gripper_qpos"],
        )
    ).astype(np.float32)
    return state

def _denormalize_action(action: torch.Tensor, processor: LeapBotWAProcessor) -> np.ndarray:
    if action.ndim == 2:
        action = action.unsqueeze(0)
    if action.ndim != 3:
        raise ValueError(f"Expected action tensor [B, T, D], got {tuple(action.shape)}")

    action_meta = processor.shape_meta["action"]
    if len(action_meta) != 1:
        raise ValueError(
            "LIBERO eval currently expects a single merged action key in shape_meta['action']."
        )

    action_key = action_meta[0]["key"]
    normalizer = processor.normalizer.normalizers["action"][action_key]
    action = action.to(dtype=torch.float32, device="cpu")
    denorm = normalizer.backward(action)
    return denorm.numpy()

def _resolve_gripper_action_encoding(cfg: DictConfig, processor: LeapBotWAProcessor) -> str:
    encoding = str(cfg.EVALUATION.get("gripper_action_encoding", "auto")).lower()
    valid_encodings = {"auto", "zero_one", "minus_one_one"}
    if encoding not in valid_encodings:
        raise ValueError(
            "EVALUATION.gripper_action_encoding must be one of "
            f"{sorted(valid_encodings)}, got {encoding!r}."
        )
    if encoding != "auto":
        return encoding

    action_meta = processor.shape_meta["action"]
    if len(action_meta) != 1:
        raise ValueError(
            "Automatic gripper encoding detection expects a single action key. "
            "Set EVALUATION.gripper_action_encoding explicitly."
        )
    action_key = action_meta[0]["key"]
    stats = processor.normalizer.normalizers["action"][action_key].stats
    gripper_min = float(torch.as_tensor(stats["min"])[-1].item())
    gripper_max = float(torch.as_tensor(stats["max"])[-1].item())

    if np.isclose(gripper_min, 0.0, atol=1e-4) and np.isclose(gripper_max, 1.0, atol=1e-4):
        return "zero_one"
    if np.isclose(gripper_min, -1.0, atol=1e-4) and np.isclose(gripper_max, 1.0, atol=1e-4):
        return "minus_one_one"
    raise ValueError(
        "Cannot infer gripper action encoding from dataset stats: "
        f"min={gripper_min}, max={gripper_max}. Set "
        "EVALUATION.gripper_action_encoding to zero_one or minus_one_one."
    )

def _get_max_steps(task_suite_name: str) -> int:
    suite_steps = {
        "libero_spatial": 400,
        "libero_object": 400,
        "libero_goal": 400,
        "libero_10": 700,
        "libero_90": 700,
    }
    if task_suite_name not in suite_steps:
        raise ValueError(f"Unknown task suite: {task_suite_name}")
    return suite_steps[task_suite_name]


def _predict_action_chunk(obs, task_description, model, processor, cfg, *,
                          action_horizon, input_w, input_h, model_device):
    views = _obs_to_per_view_jepa_input(obs, processor, model_device, model.torch_dtype)
    proprio = _normalize_proprio(_extract_sim_state(obs), processor)
    with torch.no_grad():
        prediction = model.infer_action(
            prompt=DEFAULT_PROMPT.format(task=task_description),
            input_image=views[:, 0], jepa_input_image=views,
            action_horizon=action_horizon, proprio=proprio,
            num_inference_steps=int(cfg.EVALUATION.num_inference_steps),
            sigma_shift=cfg.EVALUATION.get("sigma_shift"),
            seed=None if cfg.get("seed") is None else int(cfg.seed),
            rand_device=str(cfg.EVALUATION.get("rand_device", "cpu")),
            tiled=bool(cfg.EVALUATION.get("tiled", False)),
            negative_prompt=str(cfg.EVALUATION.get("negative_prompt", "")),
            text_cfg_scale=float(cfg.EVALUATION.get("text_cfg_scale", 1.0)),
        )
    action = _denormalize_action(prediction["action"], processor)[0]
    if _resolve_gripper_action_encoding(cfg, processor) == "zero_one":
        action[..., -1] = action[..., -1] * 2 - 1
        action = invert_gripper_action(action)
    if bool(cfg.EVALUATION.get("binarize_gripper", False)):
        action[..., -1] = np.sign(action[..., -1])
    return action, get_libero_image(obs)


def run_single_episode(env, initial_state, task_description, model, processor, cfg,
                       episode_idx, *, action_horizon, input_w, input_h, model_device):
    """Execute the reference settle/replan/step loop, returning success and video frames."""
    max_steps = _get_max_steps(cfg.EVALUATION.task_suite_name)
    replan_steps = int(cfg.EVALUATION.replan_steps)
    num_steps_wait = int(cfg.EVALUATION.num_steps_wait)
    if not 1 <= replan_steps <= action_horizon or num_steps_wait < 0:
        raise ValueError("Require 1 <= replan_steps <= action_horizon and num_steps_wait >= 0")
    env.reset()
    obs = env.set_init_state(initial_state)
    use_ensembler = bool(cfg.EVALUATION.get("use_action_ensembler", False))
    ensembler = ActionEnsembler() if use_ensembler else None
    pending_actions = []
    replay_images = []
    t = 0
    done = False
    with tqdm(total=max_steps + num_steps_wait, desc=f"Episode {episode_idx + 1}") as pbar:
        while t < max_steps + num_steps_wait:
            pbar.update(1)
            if t < num_steps_wait:
                obs, _, done, _ = env.step(get_libero_dummy_action())
                t += 1
                continue
            if not pending_actions:
                chunk, images = _predict_action_chunk(
                    obs, task_description, model, processor, cfg,
                    action_horizon=action_horizon, input_w=input_w, input_h=input_h,
                    model_device=model_device,
                )
                if ensembler is not None:
                    ensembler.add_actions(chunk, t)
                    pending_actions = [ensembler.get_action(ts).tolist() for ts in range(t, t + replan_steps)]
                else:
                    pending_actions = chunk[:replan_steps].tolist()
            else:
                images = get_libero_image(obs)
            replay_images.append(images.copy())
            obs, _, done, _ = env.step(pending_actions.pop(0))
            if done:
                break
            t += 1
    return bool(done), replay_images


def build_eval_runtime(cfg):
    if cfg.ckpt is None:
        raise ValueError("ckpt must point to a policy checkpoint")
    if not bool(cfg.model.load_text_encoder):
        raise ValueError("Simulator evaluation encodes task instructions; model.load_text_encoder must be true")
    device = _resolve_eval_device(cfg)
    model = instantiate(cfg.model, model_dtype=_mixed_precision_to_model_dtype(cfg.mixed_precision), device=device)
    model.load_checkpoint(str(cfg.ckpt))
    model = model.to(device).eval()
    stats_path = _resolve_dataset_stats_path(cfg)
    processor = instantiate(cfg.processor).eval()
    processor.set_normalizer_from_stats(load_dataset_stats_from_json(str(stats_path)))
    horizon = int(cfg.EVALUATION.action_horizon)
    if horizon <= 0:
        raise ValueError("action_horizon must be positive")
    _, height, width = processor.shape_meta["images"][0]["shape"]
    return EvalRuntime(model, processor, horizon, int(width), int(height), device, stats_path)


def run_eval_task_with_runtime(cfg, runtime, *, start_time=None):
    from libero.libero import benchmark

    started = time.time() if start_time is None else start_time
    trials = int(cfg.EVALUATION.num_trials)
    if trials <= 0:
        raise ValueError("num_trials must be positive")
    suite = benchmark.get_benchmark_dict()[cfg.EVALUATION.task_suite_name]()
    task = suite.get_task(int(cfg.EVALUATION.task_id))
    initial_states = suite.get_task_init_states(int(cfg.EVALUATION.task_id))
    if len(initial_states) == 0:
        raise ValueError("Task has no initial states")
    output = Path(cfg.EVALUATION.output_dir) / cfg.EVALUATION.task_suite_name
    output.mkdir(parents=True, exist_ok=True)
    video_dir = output / "videos"
    if cfg.EVALUATION.save_rollout_video:
        video_dir.mkdir(exist_ok=True)
    env, instruction = get_libero_env(task, LIBERO_ENV_RESOLUTION, cfg.get("seed"))
    results = dict(task_suite=cfg.EVALUATION.task_suite_name, task_id=int(cfg.EVALUATION.task_id),
                   task_description=instruction, successes=0, total_episodes=trials,
                   gpu_id=int(cfg.gpu_id), success_episodes=[], failure_episodes=[],
                   start_time=time.strftime("%Y-%m-%d %H:%M:%S"))
    try:
        for trial in range(trials):
            success, frames = run_single_episode(
                env, initial_states[trial % len(initial_states)], instruction,
                runtime.model, runtime.processor, cfg, trial,
                action_horizon=runtime.action_horizon, input_w=runtime.input_w,
                input_h=runtime.input_h, model_device=runtime.model_device,
            )
            results["successes"] += int(success)
            results["success_episodes" if success else "failure_episodes"].append(trial)
            if cfg.EVALUATION.save_rollout_video:
                save_rollout_video(video_dir, frames, f"task{cfg.EVALUATION.task_id}_trial{trial}",
                                   success=success, task_description=instruction)
    finally:
        env.close()
    results["duration"] = time.time() - started
    results["success_rate"] = results["successes"] / trials
    path = output / f"gpu{cfg.gpu_id}_task{cfg.EVALUATION.task_id}_results.json"
    path.write_text(json.dumps(results, indent=4, cls=NumpyEncoder) + "\n")
    print(f"Task {cfg.EVALUATION.task_id}: {results['successes']}/{trials} successes; results: {path}")
    return results


@hydra.main(version_base="1.3", config_path="../../configs", config_name="sim_libero")
def eval_single_process(cfg: DictConfig):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(cfg.gpu_id)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    if cfg.get("seed") is not None:
        random.seed(int(cfg.seed))
        np.random.seed(int(cfg.seed))
        torch.manual_seed(int(cfg.seed))
    output = Path(cfg.EVALUATION.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(OmegaConf.create(OmegaConf.to_container(cfg, resolve=True)), output / f"config_{cfg.EVALUATION.task_suite_name}_{cfg.EVALUATION.task_id}.yaml")
    started = time.time()
    return run_eval_task_with_runtime(cfg, build_eval_runtime(cfg), start_time=started)


if __name__ == "__main__":
    eval_single_process()
