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

import assemble_labelled_splits
from _workflow.data import (atom_metadata, digest, load_labels, structures,
                            write_charges, write_sdf)


def make_pool(root, entries):
    root.mkdir()
    sdf = root / 'sdf'
    sdf.mkdir()
    charges_dir = root / 'charges'
    charges_dir.mkdir()
    arrays, metadata, rows = {}, {}, []
    for identifier, smiles in entries:
        molecule = Chem.AddHs(Chem.MolFromSmiles(smiles))
        if AllChem.EmbedMolecule(molecule, randomSeed=42) != 0:
            raise AssertionError(f'Could not embed {identifier}')
        source = sdf / f'{identifier}.sdf'
        write_sdf(source, molecule)
        values = np.zeros(molecule.GetNumAtoms())
        arrays[identifier] = values
        metadata[identifier] = {
            'source_sdf': str(source), 'source_sha256': digest(source),
            **atom_metadata(molecule),
        }
        rows.append({
            'CSD_identifier': identifier,
            'smiles': Chem.MolToSmiles(molecule),
            'charges': json.dumps(values.tolist()),
            'atom_indices': json.dumps(metadata[identifier]['atom_indices']),
        })
    write_charges(charges_dir, rows, arrays, metadata, 'psiresp', 'all_atom')
    return sdf, charges_dir / 'charges.npz'


class AssembleLabelledSplitTests(unittest.TestCase):
    def test_exact_counts_preserve_provenance_and_disjointness(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            base_sdf, base_charges = make_pool(root / 'base', [
                ('BASEAA', 'C'), ('BASEBB', 'CO')])
            additional_sdf, additional_charges = make_pool(root / 'additional', [
                ('ADDCAA', 'CC'), ('ADDCBB', 'CN'), ('ADDCCC', 'CCC')])
            output = root / 'assembled'
            self.assertEqual(assemble_labelled_splits.main([
                '--base-sdf', str(base_sdf), '--base-charges', str(base_charges),
                '--additional-sdf', str(additional_sdf),
                '--additional-charges', str(additional_charges),
                '--output', str(output), '--train-count', '3', '--test-count', '1',
                '--seed', '42',
            ]), 0)
            split = json.loads((output / 'split.json').read_text())
            self.assertEqual((split['train_count'], split['test_count']), (3, 1))
            self.assertEqual(split['base_count'], 2)
            self.assertTrue(split['connectivity_disjoint'])
            train = structures(output / 'train' / 'sdf')
            test = structures(output / 'test' / 'sdf')
            for name, records in (('train', train), ('test', test)):
                labels, verified = load_labels(
                    output / name / 'charges' / 'charges.npz', records)
                self.assertTrue(verified)
                self.assertEqual(set(labels), {identifier for identifier, _, _ in records})
            self.assertTrue(set(split['train_ids']).isdisjoint(split['test_ids']))


if __name__ == '__main__':
    unittest.main()
