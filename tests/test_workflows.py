"""Data integrity and retrieval contracts for the standalone workflows."""

import csv
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem

import retrieve_structures
from _workflow.data import atom_metadata, digest, load_labels, read_molecule, structures, write_json, write_sdf
from _workflow.training import molecular_data
from _workflow.licensing import load_ccdc_license, VARIABLE


class WorkflowTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.mol = Chem.AddHs(Chem.MolFromSmiles("CO"))
        self.assertEqual(AllChem.EmbedMolecule(self.mol, randomSeed=42), 0)
        self.source = self.root / "sample.sdf"
        write_sdf(self.source, self.mol)

    def test_local_retrieval_preserves_atoms_and_refuses_overwrite(self):
        output = self.root / "retrieved"
        self.assertEqual(retrieve_structures.main(["--sdf", str(self.source), "--output", str(output)]), 0)
        result = read_molecule(output / "sdf/sample.sdf")
        self.assertEqual(atom_metadata(result), atom_metadata(self.mol))
        self.assertTrue(np.allclose(result.GetConformer().GetPositions(), read_molecule(self.source).GetConformer().GetPositions()))
        before = digest(output / "sdf/sample.sdf")
        with self.assertRaises(FileExistsError):
            retrieve_structures.main(["--sdf", str(self.source), "--output", str(output)])
        self.assertEqual(digest(output / "sdf/sample.sdf"), before)

    def test_csd_adapter_selects_largest_component_and_reports_partial_failure(self):
        # The licensed CSD service is unavailable locally; simulate only that boundary.
        small = SimpleNamespace(atoms=[1], rdkit=self.mol)
        large = SimpleNamespace(atoms=list(self.mol.GetAtoms()), rdkit=self.mol)
        class Reader:
            def entry(self, identifier):
                if identifier == "BAD":
                    raise KeyError(identifier)
                return SimpleNamespace(molecule=SimpleNamespace(components=[small, large]))
        class Writer:
            def __init__(self, path): self.path = path
            def __enter__(self): return self
            def __exit__(self, *args): return False
            def write(self, value):
                if value is not large:
                    raise AssertionError("Largest component was not selected")
                write_sdf(self.path, value.rdkit)
        fake = SimpleNamespace(io=SimpleNamespace(EntryReader=lambda _: Reader(), MoleculeWriter=Writer))
        source = self.root / "ids.csv"
        source.write_text("Database identifier\nGOOD\nBAD\n")
        output = self.root / "csd"
        with patch.dict(sys.modules, {"ccdc": fake}), patch.object(retrieve_structures, "load_ccdc_license"):
            code = retrieve_structures.main(["--csv", str(source), "--output", str(output)])
        self.assertEqual(code, 1)
        self.assertEqual([p.name for p in (output / "sdf").glob("*.sdf")], ["GOOD.sdf"])
        with (output / "status.csv").open() as stream:
            rows = list(csv.DictReader(stream))
        self.assertEqual([r["status"] for r in rows], ["ok", "failed"])

    def test_reference_manifest_rejects_changed_sdf_and_neural_labels(self):
        records = structures(self.source)
        q = np.zeros(self.mol.GetNumAtoms())
        labels = self.root / "charges.npz"
        np.savez_compressed(labels, sample=q)
        manifest = {"format": "ccdc-charges-v1", "charge_convention": "all_atom", "molecules": {
            "sample": {"source_sha256": digest(self.source), **atom_metadata(self.mol)}}}
        write_json(self.root / "manifest.json", manifest)
        _, verified = load_labels(labels, records)
        self.assertTrue(verified)
        self.source.write_text(self.source.read_text() + "\n")
        with self.assertRaisesRegex(ValueError, "do not match"):
            load_labels(labels, records)
        manifest["charge_convention"] = "heavy_atom_absorb_h"
        write_json(self.root / "manifest.json", manifest)
        with self.assertRaisesRegex(ValueError, "all-atom"):
            load_labels(labels, records)

    def test_labels_must_have_correct_count_and_total_charge(self):
        records = structures(self.source)
        labels = self.root / "charges.npz"
        for bad in (np.zeros(2), np.ones(self.mol.GetNumAtoms())):
            np.savez_compressed(labels, sample=bad)
            with self.assertRaises(ValueError):
                load_labels(labels, records)

    def test_hydrogen_absorption_preserves_order_and_total_charge(self):
        q = np.arange(self.mol.GetNumAtoms(), dtype=float) / 10
        q -= q.mean()
        expected = q[:2].copy()
        for atom in self.mol.GetAtoms():
            if atom.GetAtomicNum() == 1:
                expected[atom.GetNeighbors()[0].GetIdx()] += q[atom.GetIdx()]
        data = molecular_data(structures(self.source), {"sample": q})[0]
        self.assertEqual(data.h.flatten().tolist(), [6, 8])
        self.assertTrue(np.allclose(data.charges.flatten().numpy(), expected))
        self.assertAlmostEqual(data.charges.sum().item(), q.sum(), places=6)

    def test_license_loads_only_configuration_and_respects_environment(self):
        env = self.root / ".env"
        env.write_text("OTHER_VARIABLE=ignored\nexport CCDC_LICENSING_CONFIGURATION='la-code;test-key' # private\n")
        with patch.dict(os.environ, {}, clear=True):
            load_ccdc_license(env)
            self.assertEqual(os.environ[VARIABLE], "la-code;test-key")
            self.assertNotIn("OTHER_VARIABLE", os.environ)
            os.environ[VARIABLE] = "lf-server;https://example.invalid"
            load_ccdc_license(env)
            self.assertEqual(os.environ[VARIABLE], "lf-server;https://example.invalid")

    def test_license_errors_do_not_expose_values(self):
        env = self.root / ".env"
        for value in ("'la-code;test-secret", "test-secret", "'la-code;YOUR_ACTIVATION_KEY'"):
            env.write_text(f"{VARIABLE}={value}\n")
            with patch.dict(os.environ, {}, clear=True), self.assertRaises(ValueError) as caught:
                load_ccdc_license(env)
            self.assertNotIn("test-secret", str(caught.exception))


if __name__ == "__main__":
    unittest.main()
