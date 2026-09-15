import csv
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))

import build_trusted_dataset
from _workflow.data import atom_metadata, digest, structures, write_charges, write_json, write_sdf, load_labels


class TrustedDatasetTests(unittest.TestCase):
    def test_failed_and_outlier_labels_are_removed_with_hash_preservation(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / 'sdf'
            source.mkdir()
            molecule = Chem.AddHs(Chem.MolFromSmiles('CO'))
            self.assertEqual(AllChem.EmbedMolecule(molecule, randomSeed=42), 0)
            good = source / 'GOOD.sdf'
            bad = source / 'BAD.sdf'
            write_sdf(good, molecule)
            write_sdf(bad, molecule)
            records = structures(source)
            charges = root / 'charges'
            charges.mkdir()
            good_q = np.zeros(molecule.GetNumAtoms())
            bad_q = good_q.copy()
            bad_q[0], bad_q[1] = 3.0, -3.0
            arrays = {'GOOD': good_q, 'BAD': bad_q}
            metadata = {i: {'source_sdf': str(path), 'source_sha256': digest(path), **atom_metadata(mol)}
                        for i, path, mol in records}
            rows = [{'CSD_identifier': i, 'smiles': Chem.MolToSmiles(m),
                     'charges': json.dumps(arrays[i].tolist()),
                     'atom_indices': json.dumps(metadata[i]['atom_indices'])}
                    for i, _, m in records]
            write_charges(charges, rows, arrays, metadata, 'psiresp', 'all_atom')
            output = root / 'filtered'
            self.assertEqual(build_trusted_dataset.main([
                '--sdf', str(source), '--charges', str(charges), '--output', str(output),
                '--max-absolute-charge', '2']), 0)
            self.assertEqual([p.name for p in (output / 'sdf').glob('*.sdf')], ['GOOD.sdf'])
            filtered_records = structures(output / 'sdf')
            labels, verified = load_labels(output / 'charges.npz', filtered_records)
            self.assertTrue(verified)
            self.assertEqual(set(labels), {'GOOD'})
            with (output / 'status.csv').open() as stream:
                statuses = list(csv.DictReader(stream))
            self.assertEqual({row['identifier']: row['status'] for row in statuses},
                             {'GOOD': 'accepted', 'BAD': 'rejected'})
            self.assertIn('untrusted', next(row['reason'] for row in statuses if row['identifier'] == 'BAD'))
