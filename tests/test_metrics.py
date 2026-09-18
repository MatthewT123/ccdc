"""Independent checks of charge metric aggregation and units."""
from pathlib import Path
import sys
import unittest
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from _workflow import training  # Set up model imports.
from _workflow.metrics import charge_statistics
import torch


class ChargeMetricTests(unittest.TestCase):
    def test_atom_and_molecule_weighting_and_charge_conservation(self):
        # Molecule 0 has cancelling errors; molecule 1 has one net error.
        result = charge_statistics(torch.tensor([1., -1., 2.]), torch.zeros(3),
                                   torch.tensor([0, 0, 1]), torch.tensor([6, 6, 8]))
        self.assertAlmostEqual(result['mae_e'], 4/3, places=6)
        self.assertAlmostEqual(result['mse_e2'], 2.)
        self.assertAlmostEqual(result['rmse_e'], 2**0.5, places=6)
        self.assertAlmostEqual(result['molecule_mean_mse_e2'], 2.5)
        self.assertAlmostEqual(result['total_charge_mae_e'], 1.)
        self.assertAlmostEqual(result['total_charge_mse_e2'], 2.)
        self.assertEqual(result['per_element_mae_e'], {'6':1., '8':2.})

    def test_nonfinite_predictions_rejected(self):
        with self.assertRaisesRegex(ValueError, 'Non-finite'):
            charge_statistics(torch.tensor([float('nan')]), torch.zeros(1),
                              torch.tensor([0]), torch.tensor([6]))
