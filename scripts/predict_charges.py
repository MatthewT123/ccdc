"""Calculate reference RESP charges using the existing esp_generation pipeline."""

import argparse
import csv
import json
import logging
import multiprocessing
import os
from contextlib import nullcontext
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
from rdkit import Chem

from _workflow.data import atom_metadata, digest, new_output, structures, write_charges


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sdf", type=Path, required=True, help="SDF file or directory from retrieve_structures.py")
    parser.add_argument("--output", type=Path, required=True, help="New output directory")
    parser.add_argument("--max-iterations", type=int, default=3)
    parser.add_argument("--n-processes", type=int, default=1)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--precise-fit", action="store_true", help="Accurate small-system RESP linear solves; reject unconverged fits")
    args = parser.parse_args(argv)
    if args.max_iterations < 1 or args.n_processes < 1:
        parser.error("Iterations and process count must be positive")
    records = structures(args.sdf)
    output = new_output(args.output)
    # Psi4 may emit timer/scratch files relative to cwd, including at process exit.
    os.chdir(output)
    from esp_generation import ConformerRecord, RespCalculation
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
        handlers=[logging.StreamHandler(), logging.FileHandler(output / "calculation.log")])
    multiprocessing.set_start_method("spawn", force=True)
    arrays, metadata, rows, statuses = {}, {}, [], []
    for identifier, source, mol in records:
        try:
            record = ConformerRecord(identifier, mol, mol.GetNumConformers())
            from _workflow.resp_solver import precise_resp_solver
            with precise_resp_solver() if args.precise_fit else nullcontext():
                q = np.asarray(RespCalculation(record, output / "work",
                    max_iterations=args.max_iterations, n_processes=args.n_processes).run_to_completion())
            if q.shape != (mol.GetNumAtoms(),) or not np.isfinite(q).all():
                raise ValueError("Invalid charge count or non-finite charges")
            total = Chem.GetFormalCharge(mol)
            if not np.isclose(q.sum(), total, atol=1e-4):
                raise ValueError(f"Charge sum {q.sum()} differs from formal charge {total}")
            atoms = atom_metadata(mol)
            arrays[identifier] = q
            metadata[identifier] = {"source_sdf": str(source), "source_sha256": digest(source), **atoms}
            metadata[identifier]['resp_solver'] = 'dense_lstsq_rcond_1e-14' if args.precise_fit else 'psiresp_default'
            rows.append({"CSD_identifier": identifier, "smiles": Chem.MolToSmiles(mol),
                         "charges": json.dumps(q.tolist()), "atom_indices": json.dumps(atoms["atom_indices"])})
            statuses.append({"identifier": identifier, "status": "ok", "message": ""})
        except Exception as exc:
            logging.exception("%s failed", identifier)
            statuses.append({"identifier": identifier, "status": "failed", "message": str(exc)})
    write_charges(output, rows, arrays, metadata, method="psiresp", convention="all_atom")
    with (output / "status.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["identifier", "status", "message"])
        writer.writeheader()
        writer.writerows(statuses)
    failed = len(records) - len(arrays)
    print(f"Calculated RESP charges for {len(arrays)} molecules; {failed} failed. Outputs: {output}")
    return 1 if failed else 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (ValueError, OSError, ImportError) as exc:
        sys.exit(str(exc))
