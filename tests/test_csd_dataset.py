"""Scientific preparation contracts for the fixed CSD split."""

from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from create_csd_dataset import prepare


class DatasetTests(unittest.TestCase):
    def entry(self, smiles):
        mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
        self.assertEqual(AllChem.EmbedMolecule(mol, randomSeed=42), 0)
        molecule = SimpleNamespace(components=[1], to_string=lambda _: Chem.MolToMolBlock(mol),
            atoms=[SimpleNamespace(atomic_number=a.GetAtomicNum()) for a in mol.GetAtoms()])
        return SimpleNamespace(has_3d_structure=True, has_disorder=False,
            is_polymeric=False, r_factor=4.0, molecule=molecule), mol

    def test_preserves_heavy_geometry_and_generates_complete_hydrogens(self):
        entry, original = self.entry('CCCOC')
        mol, identity, count = prepare(entry, 5, 20)
        self.assertEqual(count, 5)
        self.assertEqual(mol.GetNumAtoms(), original.GetNumAtoms())
        self.assertTrue(np.allclose(mol.GetConformer().GetPositions()[:5],
            original.GetConformer().GetPositions()[:5], atol=1e-4))
        self.assertEqual(identity, Chem.MolToSmiles(Chem.RemoveHs(original), isomericSmiles=False))

    def test_rejects_charge_disorder_and_multiple_components(self):
        entry, _ = self.entry('CCCC[NH3+]')
        with self.assertRaisesRegex(ValueError, 'charged'):
            prepare(entry, 5, 20)
        entry, _ = self.entry('CCCOC')
        entry.has_disorder = True
        with self.assertRaisesRegex(ValueError, 'disorder'):
            prepare(entry, 5, 20)
        entry.has_disorder = False
        entry.molecule.components = [1, 2]
        with self.assertRaisesRegex(ValueError, 'multiple_components'):
            prepare(entry, 5, 20)

    def test_stereoisomers_share_deduplication_identity(self):
        first, _ = self.entry('CC[C@H](O)C')
        second, _ = self.entry('CC[C@@H](O)C')
        self.assertEqual(prepare(first, 5, 20)[1], prepare(second, 5, 20)[1])

    def test_rejects_poor_crystallography_and_atom_clashes(self):
        entry, mol = self.entry('CCCOC')
        entry.r_factor = 5.1
        with self.assertRaisesRegex(ValueError, 'r_factor'):
            prepare(entry, 3, 10)

        entry.r_factor = 4.0
        mol.GetConformer().SetAtomPosition(1, mol.GetConformer().GetAtomPosition(0))
        entry.molecule.to_string = lambda _: Chem.MolToMolBlock(mol)
        with self.assertRaisesRegex(ValueError, 'clash'):
            prepare(entry, 3, 10)

    def test_rejects_iodine_unsupported_by_reference_basis(self):
        entry, _ = self.entry('CCI')
        with self.assertRaisesRegex(ValueError, 'unsupported_elements'):
            prepare(entry, 3, 10)


if __name__ == '__main__':
    unittest.main()
