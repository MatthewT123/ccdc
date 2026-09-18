"""Regression checks for configured devices, model moves, and checkpoint keys."""

import copy
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "MolFLAE"))

import torch
from model.bfn4sbdd import BFN4SBDDScoreModel, BFN_charge
from model.train_loop import TrainLoop, TrainLoopCharges
from utils.config import load_config
from utils.device import resolve_device


class DeviceTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        self.env = patch.dict(os.environ, {}, clear=False)
        self.env.start()
        os.environ.pop("MOLFLAE_DEVICE", None)
        self.addCleanup(self.env.stop)
        self.cfg = load_config(ROOT / "MolFLAE/config.yaml")

    def test_selection_precedence_and_unavailable_cuda(self):
        with patch("torch.cuda.is_available", return_value=False):
            self.assertEqual(resolve_device("auto"), torch.device("cpu"))
            with self.assertRaises(ValueError):
                resolve_device("cuda")
            os.environ["MOLFLAE_DEVICE"] = "cuda"
            self.assertEqual(resolve_device("cpu", self.cfg), torch.device("cpu"))
            with self.assertRaises(ValueError):
                resolve_device(config={"runtime": {"device": "cpu"}})
            del os.environ["MOLFLAE_DEVICE"]
            with self.assertRaises(ValueError):
                resolve_device(config={"runtime": {"device": "cuda"}})
        with patch("torch.cuda.is_available", return_value=True), \
             patch("torch.cuda.current_device", return_value=1), \
             patch("torch.cuda.device_count", return_value=2):
            self.assertEqual(resolve_device("auto"), torch.device("cuda:1"))
            self.assertEqual(resolve_device("cuda:0"), torch.device("cuda:0"))
            with self.assertRaises(ValueError):
                resolve_device("cuda:2")

    def test_schedule_buffers_follow_module_moves_without_new_checkpoint_keys(self):
        for cls, key in ((BFN4SBDDScoreModel, "decoder_config"), (BFN_charge, "decoder_config_charge")):
            with self.subTest(model=cls.__name__):
                model = cls(**self.cfg[key], device="cpu")
                keys = set(model.state_dict())
                schedules = ["sigma1_coord", "beta1"]
                if cls is BFN_charge:
                    schedules.append("sigma1_charges")
                self.assertTrue(set(schedules).isdisjoint(keys))
                model.to(dtype=torch.float64)
                for name in schedules:
                    self.assertEqual(getattr(model, name).dtype, torch.float64)
                if torch.cuda.is_available():
                    device = resolve_device("cuda")
                    model.to(device)
                    self.assertEqual(model.device, device)
                    self.assertTrue(all(b.device == device for b in model.buffers()))
                    model.to("cpu")
                self.assertEqual(model.device, torch.device("cpu"))
                self.assertTrue(all(b.device.type == "cpu" for b in model.buffers()))
                self.assertEqual(set(model.state_dict()), keys)

    def test_wrappers_use_config_and_full_checkpoint_still_loads(self):
        with tempfile.TemporaryDirectory() as output:
            cfg = copy.deepcopy(self.cfg)
            cfg["evaluation"]["save_dir"] = output
            target = resolve_device("auto")
            cfg["runtime"]["device"] = str(target)
            for cls in (TrainLoop, TrainLoopCharges):
                with self.subTest(model=cls.__name__):
                    model = cls(cfg)
                    self.assertEqual(model.device, target)
                    self.assertEqual(model.decoder.device, target)
                    self.assertTrue(all(b.device == target for b in model.buffers()))
                    model.to("cpu")
                    self.assertEqual(model.decoder.device, torch.device("cpu"))
                    if cls is TrainLoop:
                        path = ROOT / "MolFLAE/ckpt-zinc9M/model-epoch=24-val_loss=3.40.ckpt"
                        if path.exists():
                            checkpoint = torch.load(path, map_location=model.device, weights_only=True)
                            model.load_state_dict(checkpoint["state_dict"], strict=True)

    def test_charge_sampling_after_device_move(self):
        target = resolve_device("auto")
        # Construct on a different device first: catches stale constructor device state.
        model = BFN_charge(**self.cfg["decoder_config_charge"], device="cpu").to(target).eval()
        n = 12
        with torch.no_grad():
            theta, samples, _ = model.sample(
                protein_pos=torch.randn(10, 3, device=target),
                protein_v=torch.randn(10, 32, device=target),
                batch_protein=torch.zeros(10, dtype=torch.long, device=target),
                batch_ligand=torch.zeros(n, dtype=torch.long, device=target),
                n_nodes=n, sample_steps=2,
            )
        for tensor in (*theta[-1], *samples[-1]):
            self.assertEqual(tensor.device, target)
            self.assertTrue(torch.isfinite(tensor).all())

    def test_charge_optimizer_has_separate_learning_rates(self):
        with tempfile.TemporaryDirectory() as output:
            cfg = copy.deepcopy(self.cfg)
            cfg["evaluation"]["save_dir"] = output
            cfg["runtime"]["device"] = "cpu"
            cfg["train"]["optimizer"]["lr"] = 1e-5
            cfg["train"]["optimizer"]["charge_lr"] = 1e-4
            model = TrainLoopCharges(cfg, device="cpu")
            model.configure_optimizers()
            groups = {group["name"]: group for group in model.optim.param_groups}
            self.assertEqual(set(groups), {"backbone", "charge_head"})
            self.assertEqual(groups["backbone"]["lr"], 1e-5)
            self.assertEqual(groups["charge_head"]["lr"], 1e-4)
            backbone_ids = {id(parameter) for parameter in groups["backbone"]["params"]}
            charge_ids = {id(parameter) for parameter in groups["charge_head"]["params"]}
            self.assertTrue(backbone_ids.isdisjoint(charge_ids))
            self.assertTrue(all(name.startswith("decoder.charge_head.")
                                for name, parameter in model.named_parameters()
                                if id(parameter) in charge_ids))


if __name__ == "__main__":
    unittest.main()
