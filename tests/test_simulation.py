import ast
import json
import tempfile
import unittest
from collections import deque
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch, MagicMock

import numpy as np
import torch
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from omegaconf import OmegaConf

from experiments.libero import eval_libero_single as libero_eval
from experiments.robotwin import eval_robotwin_single as robotwin_eval
from experiments.robotwin.leapbot_wa_policy.deploy_policy import LeapBotWARoboTwinPolicy, get_model, eval as policy_eval, reset_model

ROOT = Path(__file__).resolve().parents[1]


def sim_config(name):
    with initialize_config_dir(config_dir=str(ROOT / "configs"), version_base="1.3"):
        return compose(config_name=f"sim_{name}", overrides=["EVALUATION.output_dir=./test-output"])


def observation():
    return dict(agentview_image=np.zeros((8, 12, 3), np.uint8),
                robot0_eye_in_hand_image=np.ones((8, 12, 3), np.uint8),
                robot0_eef_pos=np.array([0.1, 0.2, 0.3]), robot0_eef_quat=np.array([0., 0., 0., 1.]),
                robot0_gripper_qpos=np.array([0.01, 0.02]))


class SimulatorTests(unittest.TestCase):
    def test_sim_configs_are_inference_only_and_enable_text(self):
        for name in ("libero", "robotwin"):
            cfg = sim_config(name)
            self.assertNotIn("data", cfg)
            self.assertNotIn("optimizer", cfg)
            self.assertTrue(cfg.model.load_text_encoder)
            self.assertNotIn("branches", cfg.model)
            self.assertNotIn("enabled", cfg.model.jepa_adapter)
            self.assertEqual(cfg.ckpt, f"weights/leapbotwa-{name}.pt")
            proc = instantiate(cfg.processor).eval()
            self.assertIn(proc.action_output_dim, cfg.model.action_dit_config.action_dims)
            self.assertGreater(cfg.EVALUATION.replan_steps, 0)

    def test_libero_runtime_builds_with_release_config(self):
        cfg = sim_config("libero")
        cfg.EVALUATION.device = "cpu"
        model = MagicMock()
        model.to.return_value = model
        model.eval.return_value = model
        processor = instantiate(cfg.processor).eval()
        stats = ROOT / "configs/stats/libero_dataset_stats.json"
        with patch.object(libero_eval, "instantiate", side_effect=[model, processor]), \
             patch.object(libero_eval, "_resolve_dataset_stats_path", return_value=stats):
            runtime = libero_eval.build_eval_runtime(cfg)
        self.assertIs(runtime.model, model)
        self.assertIs(runtime.processor, processor)
        model.load_checkpoint.assert_called_once_with(cfg.ckpt)
        self.assertEqual(runtime.action_horizon, cfg.EVALUATION.action_horizon)

    def test_libero_settle_replan_and_action_queue(self):
        cfg = sim_config("libero")
        cfg.EVALUATION.num_steps_wait = 2
        cfg.EVALUATION.replan_steps = 2
        actions = []
        env = SimpleNamespace(reset=lambda: None, set_init_state=lambda state: observation())
        def step(action):
            actions.append(action)
            return observation(), 0, len(actions) == 5, {}
        env.step = step
        chunk = np.arange(28, dtype=np.float32).reshape(4, 7)
        with patch.object(libero_eval, "_predict_action_chunk", return_value=(chunk, {"image": np.zeros((2,2,3))})) as predict:
            done, frames = libero_eval.run_single_episode(env, None, "task", None, None, cfg, 0,
                action_horizon=4, input_w=224, input_h=224, model_device="cpu")
        self.assertTrue(done)
        self.assertEqual(predict.call_count, 2)
        self.assertEqual(actions[:2], [[0,0,0,0,0,0,-1]] * 2)
        np.testing.assert_array_equal(actions[2:], chunk[[0,1,0]])
        self.assertEqual(len(frames), 3)

    def test_libero_gripper_transform_and_model_inputs(self):
        cfg = sim_config("libero")
        cfg.EVALUATION.gripper_action_encoding = "zero_one"
        cfg.EVALUATION.binarize_gripper = True
        proc = instantiate(cfg.processor).eval()
        proc.set_normalizer_from_stats({kind:{"default":{"global_min":torch.zeros(dim), "global_max":torch.ones(dim)}} for kind,dim in [("action",7),("state",8)]})
        recorded = {}
        class Model:
            torch_dtype = torch.float32
            def infer_action(self, **kwargs):
                recorded.update(kwargs)
                out = torch.zeros(2,7)
                out[:, -1] = torch.tensor([-1., 1.])
                return {"action":out}
        actions,_ = libero_eval._predict_action_chunk(observation(), "pick cup", Model(), proc, cfg,
            action_horizon=2,input_w=224,input_h=224,model_device="cpu")
        np.testing.assert_array_equal(actions[:, -1], [1., -1.])
        self.assertEqual(recorded["prompt"], "A video recorded from a robot's point of view executing the following instruction: pick cup")
        self.assertEqual(tuple(recorded["jepa_input_image"].shape), (1,2,3,224,224))
        self.assertEqual(tuple(recorded["proprio"].shape), (1,8))

    def test_libero_writes_results_and_closes_environment(self):
        cfg = sim_config("libero")
        cfg.EVALUATION.num_trials=3
        cfg.EVALUATION.save_rollout_video=False
        suite=SimpleNamespace(get_task=lambda idx: object(), get_task_init_states=lambda idx:["state0","state1"])
        benchmark=SimpleNamespace(get_benchmark_dict=lambda:{"libero_spatial":lambda:suite})
        fake_module=SimpleNamespace(benchmark=benchmark)
        closed=[]
        env=SimpleNamespace(close=lambda:closed.append(True))
        runtime=libero_eval.EvalRuntime(None,None,32,224,224,"cpu",Path("stats.json"))
        with tempfile.TemporaryDirectory() as tmp:
            cfg.EVALUATION.output_dir=tmp
            with patch.dict("sys.modules", {"libero":SimpleNamespace(libero=fake_module),"libero.libero":fake_module}), \
                 patch.object(libero_eval,"get_libero_env",return_value=(env,"task")), \
                 patch.object(libero_eval,"run_single_episode",side_effect=[(True,[]),(False,[]),(True,[])]) as episode:
                result=libero_eval.run_eval_task_with_runtime(cfg,runtime)
            payload=json.loads((Path(tmp)/"libero_spatial/gpu0_task0_results.json").read_text())
            self.assertEqual(payload["successes"],2)
            self.assertEqual(result["success_rate"],2/3)
            self.assertEqual([call.args[1] for call in episode.call_args_list],["state0","state1","state0"])
        self.assertEqual(closed,[True])

    def test_robotwin_queue_uses_qpos_and_resets(self):
        policy=LeapBotWARoboTwinPolicy.__new__(LeapBotWARoboTwinPolicy)
        policy.pending_actions=deque();policy.pending_action_context=deque()
        policy.replan_steps=2;policy.episode_count=0;policy.step_count=0;policy.inference_count=0
        policy._last_inference_record={};policy.timing_enabled=False
        policy._timing_rollout={"infer_s":0.,"sim_s":0.}
        chunk=np.arange(42,dtype=np.float32).reshape(3,14)
        def infer(**kwargs):
            policy._last_inference_record={"inference_id":policy.inference_count}
            policy.inference_count+=1
            return chunk
        policy._infer_action_chunk=infer
        sent=[]
        env=SimpleNamespace(get_instruction=lambda:"task",take_action=lambda action,action_type:sent.append((action,action_type)))
        policy_eval(env,policy,{})
        self.assertFalse(policy.should_request_observation())
        policy_eval(env,policy,None)
        self.assertTrue(policy.should_request_observation())
        policy_eval(env,policy,{})
        np.testing.assert_array_equal([a for a,_ in sent],chunk[[0,1,0]])
        self.assertEqual([kind for _,kind in sent],["qpos"]*3)
        self.assertEqual(policy.inference_count,2)
        reset_model(policy)
        self.assertTrue(policy.should_request_observation())
        self.assertEqual(policy.step_count,0)

    def test_robotwin_command_forwards_episode_and_replan_settings(self):
        cfg=sim_config("robotwin");cfg.EVALUATION.task_name="click_alarmclock"
        cmd=robotwin_eval.build_command(cfg,snapshot=Path('/tmp/config with space.yaml'),output_dir=Path('/tmp/out'),checkpoint=Path('/tmp/model.pt'),dataset_stats=Path('/tmp/stats.json'))
        options=dict(zip(cmd[cmd.index('--overrides')+1::2],cmd[cmd.index('--overrides')+2::2]))
        self.assertEqual(ast.literal_eval(options['--sim_cfg_path']),'/tmp/config with space.yaml')
        self.assertEqual(options['--eval_num_episodes'],'50')
        self.assertEqual(options['--replan_steps'],'24')
        self.assertEqual(ast.literal_eval(options['--policy_name']),'leapbot_wa_policy')

    def test_robotwin_launcher_preserves_asset_paths_across_cwd(self):
        cfg = sim_config("robotwin")
        cfg.EVALUATION.task_name = "click_alarmclock"
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            simulator = base / "simulator"
            (simulator / "script").mkdir(parents=True)
            (simulator / "script/eval_policy.py").write_text("# Simulator stand-in")
            checkpoint = base / "model.pt"
            checkpoint.touch()
            stats = base / "stats.json"
            stats.write_text("{}")
            cfg.ckpt = str(checkpoint)
            cfg.EVALUATION.dataset_stats_path = str(stats)
            cfg.EVALUATION.robotwin_root = str(simulator)
            cfg.EVALUATION.output_dir = str(base / "results")
            cfg.model.jepa_supervision.vjepa_checkpoint = "weights/vjepa.pt"
            cfg.model.jepa_adapter.checkpoint_path = "weights/adapter.pt"
            with patch.object(robotwin_eval.subprocess, "Popen") as start, \
                 patch.dict("os.environ", {"LEAPBOT_WA_TEXT_ENCODER": "weights/text.pt"}):
                process = start.return_value.__enter__.return_value
                process.stdout = iter(["simulator log\n"])
                process.wait.return_value = 0
                robotwin_eval.main.__wrapped__(cfg)
                launched = start.call_args.kwargs
                self.assertEqual(launched["cwd"], simulator)
                self.assertEqual(launched["env"]["LEAPBOT_WA_TEXT_ENCODER"], str(ROOT / "weights/text.pt"))
                self.assertIn(str(ROOT / "experiments/robotwin"), launched["env"]["PYTHONPATH"])
            saved = OmegaConf.load(base / "results/click_alarmclock/sim_config.yaml")
            self.assertEqual(saved.model.jepa_supervision.vjepa_checkpoint, str(ROOT / "weights/vjepa.pt"))
            self.assertEqual(saved.model.jepa_adapter.checkpoint_path, str(ROOT / "weights/adapter.pt"))
            self.assertEqual((base / "results/click_alarmclock/eval.log").read_text(), "simulator log\n")

    def test_robotwin_get_model_uses_saved_inference_config(self):
        cfg=sim_config("robotwin")
        with tempfile.TemporaryDirectory() as tmp:
            config=Path(tmp)/'sim.yaml';stats=Path(tmp)/'stats.json'
            OmegaConf.save(cfg,config);stats.write_text('{}')
            with patch('experiments.robotwin.leapbot_wa_policy.deploy_policy.LeapBotWARoboTwinPolicy') as ctor:
                get_model(dict(sim_cfg_path=str(config),ckpt_setting='policy.pt',dataset_stats_path=str(stats),device='cpu'))
                args=ctor.call_args.kwargs
                self.assertEqual(args['action_horizon'],32)
                self.assertEqual(args['replan_steps'],24)
                self.assertEqual(args['camera_view_keys'],('cam_high','cam_left_wrist','cam_right_wrist'))
                self.assertEqual(args['processor_cfg']['proprio_output_dim'],14)


if __name__ == '__main__':
    unittest.main()
