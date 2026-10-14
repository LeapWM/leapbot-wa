import tempfile
import unittest
from pathlib import Path
from types import MethodType

import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf

from leapbot_wa.models.wan22.leapbot_wa import LeapBotWA
from leapbot_wa.models.wan22.action_dit import ActionDiT
from leapbot_wa.models.wan22.jepa_state_expert import JEPAMoTStateExpert
from leapbot_wa.models.wan22.mot import MoT

ROOT = Path(__file__).resolve().parents[1]


def make_tiny_model(model_class=LeapBotWA, action_dim=7):
    torch.manual_seed(17)
    action = ActionDiT(hidden_dim=16, action_dim=action_dim, action_dims=[7, 14], ffn_dim=32,
                       text_dim=16, freq_dim=8, eps=1e-6, num_heads=2, attn_head_dim=8, num_layers=2)
    expert = JEPAMoTStateExpert(vjepa_dim=8, hidden_dim=16, ffn_dim=32,
                       text_dim=16, freq_dim=8, num_heads=2, attn_head_dim=8, num_layers=2,
                       image_size=32, patch_size=16, num_view_slots=3)
    mixtures = {"jepa_state": expert, "action": action}
    # The frozen encoder is replaced with deterministic features; attention and denoising are real.
    model = model_class(video_expert=None, action_expert=action, mot=MoT(mixtures, False), vae=torch.nn.Identity(),
                        text_dim=16, proprio_dim=8, device="cpu", torch_dtype=torch.float32,
                        vjepa_online_encoder=torch.nn.Identity(), jepa_state_expert=expert,
                        jepa_supervision_config={"enabled": True},
                        branch_config={"video": False, "jepa": True, "action": True})
    def current(self, image):
        tokens = torch.arange(64, dtype=torch.float32).reshape(1, 1, 8, 8) / 64 + image.mean()
        return {"latents": tokens, "raw_latents": tokens, "grid_size": (2, 4),
                "view_ids": torch.tensor([0]*4+[1]*4)}
    model._encode_jepa_current_from_input_image = MethodType(current, model)
    return model.eval()


def model_inputs(action_dim=7):
    return dict(prompt=None, input_image=torch.zeros(1, 3, 32, 32),
                jepa_input_image=torch.zeros(1, 2, 3, 32, 32),
                context=torch.arange(64, dtype=torch.float32).reshape(4, 16)/64,
                context_mask=torch.tensor([True, True, True, False]), proprio=torch.zeros(1, 8),
                action_horizon=4, action_dim=action_dim, seed=42, num_inference_steps=3)


class InferenceTests(unittest.TestCase):
    def test_expert_path_is_deterministic_and_checkpoint_compatible(self):
        for action_dim in (7, 14):
            with self.subTest(action_dim=action_dim):
                model = make_tiny_model(action_dim=action_dim)
                args = model_inputs(action_dim)
                actual = model.infer_action(**args)["action"]
                self.assertEqual(tuple(actual.shape), (4, action_dim))
                self.assertTrue(torch.isfinite(actual).all())
                torch.testing.assert_close(actual, model.infer_action(**args)["action"], rtol=0, atol=0)
                payload = {"mot": model.mot.state_dict(), "proprio_encoder": model.proprio_encoder.state_dict(),
                           "branches": model.branch_config}
                with tempfile.TemporaryDirectory() as tmp:
                    path = Path(tmp)/"model.pt"
                    torch.save(payload, path)
                    restored = make_tiny_model(action_dim=action_dim)
                    restored.load_checkpoint(path)
                    torch.testing.assert_close(actual, restored.infer_action(**args)["action"], rtol=0, atol=0)

    def test_component_paths_are_relative_to_repository(self):
        import os
        from leapbot_wa.models.wan22.jepa_adapter import resolve_checkpoint_path
        original = Path.cwd()
        with tempfile.TemporaryDirectory() as tmp:
            try:
                os.chdir(tmp)
                for name in ("adapter_latest.pt", "vjepa_latest.pt"):
                    self.assertEqual(resolve_checkpoint_path(f"weights/{name}"), ROOT / "weights" / name)
                absolute = Path(tmp) / "external.pt"
                self.assertEqual(resolve_checkpoint_path(absolute), absolute)
            finally:
                os.chdir(original)

    def test_partial_checkpoint_is_rejected(self):
        model = make_tiny_model()
        payload = {"mot": dict(model.mot.state_dict()),
                   "proprio_encoder": model.proprio_encoder.state_dict()}
        payload["mot"].pop(next(iter(payload["mot"])))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "partial.pt"
            torch.save(payload, path)
            with self.assertRaises(RuntimeError):
                model.load_checkpoint(path)

    def test_release_rejects_adapter_ablation(self):
        cfg = OmegaConf.load(ROOT / "configs/libero.yaml")
        cfg.model.jepa_adapter.enabled = False
        with self.assertRaisesRegex(Exception, "fixed by the released architecture"):
            instantiate(cfg.model, device="cpu")

    def test_configs_resolve_processors_without_training_code(self):
        for path in (ROOT/"configs").glob("*.yaml"):
            if path.name.startswith("sim_"):
                continue
            cfg = OmegaConf.load(path)
            processor = instantiate(cfg.processor).eval()
            self.assertIn(processor.action_output_dim, cfg.model.action_dit_config.action_dims)
            self.assertEqual(processor.num_output_cameras, len(processor.shape_meta["images"]))
            self.assertNotIn("data", cfg)




    def test_action_only_loader_does_not_require_wan_assets(self):
        from leapbot_wa.models.wan22.helpers.loader import load_wan22_ti2v_5b_components
        cfg = OmegaConf.load(ROOT/"configs/libero.yaml")
        components = load_wan22_ti2v_5b_components(
            device="cpu", torch_dtype=torch.float32,
            dit_config=OmegaConf.to_container(cfg.model.video_dit_config, resolve=True),
            load_dit=False, load_text_encoder=False, skip_dit_load_from_pretrain=True,
        )
        self.assertIsNone(components.dit)
        self.assertIsNone(components.vae)
        self.assertIsNone(components.text_encoder)



if __name__ == "__main__":
    unittest.main()
