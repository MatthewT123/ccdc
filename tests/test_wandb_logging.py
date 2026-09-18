import sys
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))

from finetune import wandb_diagnostic_metrics, wandb_training_metrics


class WandbLoggingTests(unittest.TestCase):
    def test_diagnostics_select_only_two_losses_per_split(self):
        result = {
            'training_set': {'objectives': {'structure_loss': 1.25, 'charge_loss': 2.5,
                                            'coordinate_loss': 9, 'kl_loss': -1}},
            'test_set': {'objectives': {'structure_loss': 3.0, 'charge_loss': 4.0}},
        }
        self.assertEqual(wandb_diagnostic_metrics(result), {
            'evaluation/training_set/reconstruction_loss': 1.25,
            'evaluation/training_set/charge_prediction_loss': 2.5,
            'evaluation/test_set/reconstruction_loss': 3.0,
            'evaluation/test_set/charge_prediction_loss': 4.0,
        })

    def test_training_selector_omits_gradient_and_other_components(self):
        self.assertEqual(wandb_training_metrics({
            'structure_loss': 1.0, 'charge_loss': 2.0,
            'gradient_norm': 3.0, 'lr': 0.001,
        }), {'train/reconstruction_loss': 1.0, 'train/charge_prediction_loss': 2.0})
