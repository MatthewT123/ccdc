import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))

import split_labelled_dataset
from _workflow.data import (atom_metadata, digest, load_labels, structures,
                            write_charges, write_sdf)


class LabelledSplitTests(unittest.TestCase):
    def test_split_preserves_labels_and_is_disjoint(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / 'sdf'
            source.mkdir()
            smiles = {'A': 'C', 'B': 'CO', 'C': 'CC', 'D': 'CN'}
            arrays = {}
            metadata = {}
            rows = []
            for identifier, smiles_string in smiles.items():
                molecule = Chem.AddHs(Chem.MolFromSmiles(smiles_string))
                self.assertEqual(AllChem.EmbedMolecule(molecule, randomSeed=42), 0)
                path = source / f'{identifier}.sdf'
                write_sdf(path, molecule)
                values = np.zeros(molecule.GetNumAtoms())
                arrays[identifier] = values
                metadata[identifier] = {
                    'source_sdf': str(path), 'source_sha256': digest(path),
                    **atom_metadata(molecule),
                }
                rows.append({
                    'CSD_identifier': identifier,
                    'smiles': Chem.MolToSmiles(molecule),
                    'charges': json.dumps(values.tolist()),
                    'atom_indices': json.dumps(metadata[identifier]['atom_indices']),
                })
            charges = root / 'charges'
            charges.mkdir()
            write_charges(charges, rows, arrays, metadata, 'psiresp', 'all_atom')
            output = root / 'split'

            self.assertEqual(split_labelled_dataset.main([
                '--sdf', str(source), '--charges', str(charges / 'charges.npz'),
                '--output', str(output), '--train-fraction', '0.5', '--seed', '42',
            ]), 0)
            split = json.loads((output / 'split.json').read_text())
            self.assertEqual((split['train_count'], split['test_count']), (2, 2))
            self.assertTrue(set(split['train_ids']).isdisjoint(split['test_ids']))
            for name in ('train', 'test'):
                records = structures(output / name / 'sdf')
                labels, verified = load_labels(
                    output / name / 'charges' / 'charges.npz', records)
                self.assertTrue(verified)
                self.assertEqual(set(labels), {identifier for identifier, _, _ in records})


if __name__ == '__main__':
    unittest.main()
