"""Launch RoboTwin's simulator evaluation with the LeapBot-WA policy plugin."""
import os
from pathlib import Path
import subprocess
import sys

import hydra
from omegaconf import DictConfig, OmegaConf

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _resolve_path(value):
    path = Path(os.path.expandvars(os.path.expanduser(str(value))))
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def build_command(cfg, *, snapshot, output_dir, checkpoint, dataset_stats):
    """Use RoboTwin's --config/--overrides API; argv is never passed to a shell."""
    policy_dir = PROJECT_ROOT / "experiments/robotwin/leapbot_wa_policy"
    options = dict(
        task_name=cfg.EVALUATION.task_name, task_config=cfg.EVALUATION.task_config,
        ckpt_setting=str(checkpoint), seed=cfg.seed, policy_name="leapbot_wa_policy",
        instruction_type=cfg.EVALUATION.instruction_type,
        eval_num_episodes=cfg.EVALUATION.eval_num_episodes,
        sim_cfg_path=str(snapshot), eval_output_dir=str(output_dir),
        mixed_precision=cfg.mixed_precision, device=cfg.EVALUATION.device,
        dataset_stats_path=str(dataset_stats),
    )
    for key in ("action_horizon", "replan_steps", "num_inference_steps", "sigma_shift",
                "text_cfg_scale", "negative_prompt", "rand_device", "tiled",
                "timing_enabled", "skip_get_obs_within_replan", "expert_check"):
        options[key] = cfg.EVALUATION.get(key)
    cmd = [sys.executable, "-u", "script/eval_policy.py", "--config", str(policy_dir / "deploy_policy.yml"), "--overrides"]
    for key, value in options.items():
        if value is not None:
            formatted = repr(value) if isinstance(value, str) else str(value)
            cmd.extend([f"--{key}", formatted])
    return cmd


@hydra.main(version_base="1.3", config_path="../../configs", config_name="sim_robotwin")
def main(cfg: DictConfig):
    if cfg.ckpt is None or cfg.EVALUATION.task_name is None:
        raise ValueError("ckpt and EVALUATION.task_name are required")
    if int(cfg.EVALUATION.eval_num_episodes) <= 0:
        raise ValueError("eval_num_episodes must be positive")
    if str(cfg.EVALUATION.policy_name) != "leapbot_wa_policy":
        raise ValueError("policy_name must be leapbot_wa_policy")
    checkpoint = _resolve_path(cfg.ckpt)
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    robotwin_root = _resolve_path(cfg.EVALUATION.robotwin_root)
    if not (robotwin_root / "script/eval_policy.py").is_file():
        raise FileNotFoundError(f"RoboTwin evaluation script missing: {robotwin_root}")
    explicit_stats = cfg.EVALUATION.dataset_stats_path
    candidates = [_resolve_path(explicit_stats)] if explicit_stats else [p / "dataset_stats.json" for p in list(checkpoint.parents)[:4]]
    stats = next((p for p in candidates if p.is_file()), None)
    if stats is None:
        raise FileNotFoundError("Set EVALUATION.dataset_stats_path to the matching statistics")
    output = _resolve_path(cfg.EVALUATION.output_dir) / str(cfg.EVALUATION.task_name)
    output.mkdir(parents=True, exist_ok=True)
    snapshot = output / "sim_config.yaml"
    resolved = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
    # The child runs in the simulator checkout, not the model repository.
    for key in ("model.jepa_supervision.vjepa_checkpoint", "model.jepa_supervision.vjepa2_root",
                "model.jepa_adapter.checkpoint_path", "model.action_dit_pretrained_path",
                "model.jepa_expert.pretrained_path", "model.jepa_expert.backbone_pretrained_path",
                "model.jepa_expert.jepa_expert_pretrained_path"):
        value = OmegaConf.select(resolved, key)
        if value:
            OmegaConf.update(resolved, key, str(_resolve_path(value)))
    OmegaConf.save(resolved, snapshot)
    cmd = build_command(cfg, snapshot=snapshot, output_dir=output, checkpoint=checkpoint, dataset_stats=stats)
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(cfg.gpu_id)
    env["PYTHONUNBUFFERED"] = "1"
    for key in ("LEAPBOT_WA_TEXT_ENCODER", "LEAPBOT_WA_TOKENIZER", "LEAPBOT_WA_VJEPA_ROOT"):
        if env.get(key):
            env[key] = str(_resolve_path(env[key]))
    env["PYTHONPATH"] = os.pathsep.join([str(PROJECT_ROOT), str(PROJECT_ROOT / "experiments/robotwin"), env.get("PYTHONPATH", "")])
    log_path = output / "eval.log"
    with log_path.open("w") as log:
        with subprocess.Popen(cmd, cwd=robotwin_root, env=env, stdout=subprocess.PIPE,
                              stderr=subprocess.STDOUT, text=True, bufsize=1) as process:
            for line in process.stdout:
                sys.stdout.write(line)
                log.write(line)
                log.flush()
            code = process.wait()
    if code:
        raise RuntimeError(f"RoboTwin evaluation exited with {code}; see {log_path}")
    print(f"RoboTwin evaluation completed. Results and log: {output}")


if __name__ == "__main__":
    main()
