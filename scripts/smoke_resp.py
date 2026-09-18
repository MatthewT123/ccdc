"""Exercise the real RESP CLI on water and verify its saved outputs."""

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem

ROOT = Path(__file__).resolve().parents[1]


def main():
    output_root = ROOT / "runs"
    output_root.mkdir(exist_ok=True)
    output = Path(tempfile.mkdtemp(prefix="resp-", dir=output_root))
    inputs = output / "input"
    inputs.mkdir()
    mol = Chem.AddHs(Chem.MolFromSmiles("O"))
    assert AllChem.EmbedMolecule(mol, randomSeed=42) == 0
    with Chem.SDWriter(str(inputs / "water.sdf")) as writer:
        writer.write(mol)
    env = dict(os.environ, OMP_NUM_THREADS="2", MKL_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2")
    command = [
        sys.executable, str(ROOT / "esp_generation.py"), str(inputs),
        "--working-dir", str(output / "work"),
        "--output-csv", str(output / "charges.csv"),
        "--output-npz", str(output / "charges.npz"),
        "--verbose",
    ]
    print(f"Running water RESP calculation; outputs: {output}", flush=True)
    with (output / "cli.log").open("w") as log:
        subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT, timeout=300, check=True)
    with np.load(output / "charges.npz", allow_pickle=False) as archive:
        assert archive.files == ["water"], f"RESP did not produce water charges; inspect {output / 'cli.log'}"
        charges = archive["water"]
    assert charges.shape == (3,) and np.isfinite(charges).all()
    assert abs(charges.sum()) < 1e-5
    frame = pd.read_csv(output / "charges.csv")
    assert len(frame) == 1 and frame.iloc[0]["CSD_identifier"] == "water"
    assert np.allclose(json.loads(frame.iloc[0]["resp_charges"]), charges)
    result = {"molecule": "water", "charges": charges.tolist(), "total_charge": float(charges.sum()), "output": str(output)}
    (output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
