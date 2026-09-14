"""Strict molecular I/O with explicit atom order and output provenance."""

import csv
import hashlib
import json
from pathlib import Path
import re

import numpy as np
from rdkit import Chem


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def new_output(path):
    path = Path(path).resolve()
    path.mkdir(parents=True, exist_ok=False)
    return path


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def valid_identifier(value):
    value = str(value).strip()
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", value):
        raise ValueError(f"Unsafe or empty molecule identifier: {value!r}")
    return value


def sdf_paths(source):
    source = Path(source).resolve()
    paths = [source] if source.is_file() else sorted(source.glob("*.sdf"))
    if not paths or any(p.suffix.lower() != ".sdf" for p in paths):
        raise ValueError(f"Expected an SDF file or a directory containing SDF files: {source}")
    return paths


def read_molecule(path):
    with Chem.SDMolSupplier(str(path), removeHs=False) as supplier:
        molecules = list(supplier)
    if len(molecules) != 1 or molecules[0] is None:
        raise ValueError(f"Expected exactly one parseable molecule in {path}")
    mol = molecules[0]
    if not mol.GetNumConformers() or not mol.GetConformer().Is3D():
        raise ValueError(f"A 3D conformer is required: {path}")
    if not np.isfinite(mol.GetConformer().GetPositions()).all():
        raise ValueError(f"Non-finite coordinates in {path}")
    return mol


def structures(source):
    return [(valid_identifier(p.stem), p, read_molecule(p)) for p in sdf_paths(source)]


def write_sdf(path, mol):
    with Chem.SDWriter(str(path)) as writer:
        writer.write(mol)


def atom_metadata(mol, indices=None):
    if indices is None:
        indices = list(range(mol.GetNumAtoms()))
    return {"atom_indices": list(indices),
            "atomic_numbers": [mol.GetAtomWithIdx(i).GetAtomicNum() for i in indices]}


def write_charges(output, rows, arrays, metadata, method, convention):
    with (output / "charges.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["CSD_identifier", "smiles", "charges", "atom_indices"])
        writer.writeheader()
        writer.writerows(rows)
    np.savez_compressed(output / "charges.npz", **arrays)
    write_json(output / "manifest.json", {
        "format": "ccdc-charges-v1", "method": method, "charge_convention": convention,
        "molecules": metadata,
    })


def load_labels(path, records):
    path = Path(path).resolve()
    manifest_path = path.with_name("manifest.json")
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if manifest.get("format") != "ccdc-charges-v1" or manifest.get("charge_convention") != "all_atom":
            raise ValueError("Fine-tuning requires all-atom reference charges, not model predictions.")
        metadata = manifest["molecules"]
        for identifier, sdf, mol in records:
            entry = metadata.get(identifier, {})
            if entry.get("source_sha256") != digest(sdf) or entry.get("atomic_numbers") != atom_metadata(mol)["atomic_numbers"]:
                raise ValueError(f"Reference charges do not match the source SDF: {identifier}")
    else:
        metadata = None  # Legacy NPZ: atom counts are verifiable, provenance is not.
    with np.load(path, allow_pickle=False) as archive:
        missing = {i for i, _, _ in records} - set(archive.files)
        if missing:
            raise ValueError(f"Missing charge labels: {sorted(missing)}")
        labels = {i: np.asarray(archive[i], dtype=np.float64) for i, _, _ in records}
    for identifier, _, mol in records:
        q = labels[identifier]
        if q.shape != (mol.GetNumAtoms(),) or not np.isfinite(q).all():
            raise ValueError(f"Charge count or values invalid for {identifier}")
        if not np.isclose(q.sum(), Chem.GetFormalCharge(mol), atol=1e-4):
            raise ValueError(f"Reference charge sum differs from formal charge for {identifier}")
    return labels, metadata is not None
