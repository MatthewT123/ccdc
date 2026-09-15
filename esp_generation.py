
#data structure
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, Optional
import logging
import multiprocessing

#input output
import argparse
import json


#functionality
import numpy as np
import pandas as pd
import psiresp
from rdkit import Chem
import subprocess

# to log whether each molecule from directory has been processed
# without interfering with the standard output of rdkit, psi4, and psiresp
logger = logging.getLogger('esp_generation')

# ---------------------------------------------------------------------------
# Step 1: load molecules from a directory of SDF files
# input file name structure within that directory: 'CSD database identifier'.sdf
# ---------------------------------------------------------------------------

def mol_load(path):
    with Chem.SDMolSupplier(str(path), removeHs=False) as suppl:
        mol = next(iter(suppl), None)
    return mol

@dataclass
class ConformerRecord:
    CSD_identifier: str
    mol: Chem.Mol
    n_conformers: int

    @property
    def has_conformer(self) -> bool: #except true/false output
        return self.n_conformers > 0
    
class sdfLoader:
    '''Read *.sdf in input directory and loads each as a ConformerRecord, which include:
        CSD_identifier
        mol
        n_conformers (no. of conformers in that sdf)
    '''

    def __init__(self, sdf_dir: Path):
        self.sdf_dir = Path(sdf_dir)
    
    def load(self) -> Iterator[ConformerRecord]:
        sdf_paths = sorted(self.sdf_dir.glob("*.sdf"))
        logger.info("Found %d SDF file(s) in %s", len(sdf_paths), self.sdf_dir)

        for path in sdf_paths:
            CSD_identifier = path.stem #prefix of the path
            
            mol = mol_load(path)
            if mol is None:
                logger.warning("[%s] could not be parsed from %s", CSD_identifier, path)
                continue

            n_conformers = mol.GetNumConformers()
            if n_conformers == 0:
                logger.warning("[%s] no conformers found, skipping", CSD_identifier)
            else:
                logger.info("[%s] %d conformer(s) found", CSD_identifier, n_conformers)
            
            yield ConformerRecord(CSD_identifier=CSD_identifier, 
                                  mol=mol, 
                                  n_conformers=n_conformers)


# ----------------------------------------------------------------------------------------------
# Step 2: running the psiresp-generated file in bash
# ----------------------------------------------------------------------------------------------

class Psi4Runner:
    ''' running the psiresp-generated run_*.sh and log its output'''

    def run(self, script_path: Path) -> None:
        logger.info("Running %s", script_path)
        result = subprocess.run(
            ["bash", script_path.name],
            cwd=script_path.parent,
            capture_output=True,
            text=True
        )

        log_path = script_path.with_suffix(".log")
        log_path.write_text(
            f"$ bash {script_path.name}\n\n--- stdout ---\n{result.stdout}\n\n--- stderr ---\n{result.stderr}\n"
        )

        if result.returncode != 0:
            tail = "\n".join(result.stderr.strip().splitlines()[-10:])
            raise RuntimeError(
                f"{script_path} exited with code {result.returncode}; "
                f"see {log_path} for full output. Last stderr lines:\n{tail}"
            )

# ---------------------------------------------------------------------------
# Steps 3: build and drive one molecule's psiresp job to completion
# ---------------------------------------------------------------------------

class RespCalculation:
    """Owns one molecule's psiresp Job: build it, drive it through repeated
    SystemExit/script/rerun cycles (one per psiresp stage: optimization, then
    single_point), and return the final RESP charges."""

    def __init__(self, record: ConformerRecord, working_directory: Path,
                 script_runner: Optional[Psi4Runner] = None, max_iterations: int = 3,
                 n_processes: int = 1, grid_radii: Optional[dict[str, float]] = None):
        self.record = record
        self.working_directory = Path(working_directory) / f"{record.CSD_identifier}"
        self.script_runner = script_runner or Psi4Runner()
        self.max_iterations = max_iterations
        self.n_processes = n_processes
        self.grid_radii = dict(grid_radii or {})
        self._executed_scripts: set[Path] = set()

    def _build_job(self) -> psiresp.Job:
        '''to build the psiresp object'''
        psiresp_mol = psiresp.Molecule.from_rdkit(self.record.mol)
        job = psiresp.Job(molecules=[psiresp_mol], working_directory=self.working_directory,
                          n_processes=self.n_processes)
        job.grid_options.vdw_radii.update(self.grid_radii)
        return job
    
    def _run_pending_scripts(self) -> bool:
        # Each psiresp stage (optimization/, single_point/) writes its own
        # run_<jobname>.sh; only run ones we haven't executed yet.
        pending = [
            p for p in self.working_directory.glob("*/run_*.sh")
            if p not in self._executed_scripts
        ]

        for script_path in pending:
            self.script_runner.run(script_path)
            self._executed_scripts.add(script_path)
        return bool(pending)
    
    def run_to_completion(self) -> np.ndarray:
        job = self._build_job()

        for iteration in range(1, self.max_iterations + 1):
            try:
                job.run(client=None)
            except SystemExit:
                logger.info("[%s] iteration %d: QM computation required", self.record.CSD_identifier, iteration)

                if not self._run_pending_scripts():
                    raise RuntimeError(
                        f"[{self.record.CSD_identifier}] psiresp requested more QM computation, "
                        "but no new run_*.sh script was found"
                    )
                continue
            else:
                return job.charges[0] #numpy array of atom charges for the first molecule

        raise RuntimeError(f"[{self.record.CSD_identifier}] did not converge within {self.max_iterations} iterations")
    
    # ---------------------------------------------------------------------------
# Step 4: collect and write results
# ---------------------------------------------------------------------------

@dataclass
class RespResultCollector:
    rows: list = field(default_factory=list)
    charge_arrays: dict = field(default_factory=dict)

    def add(self, record: ConformerRecord, charges: np.ndarray) -> None:
        smiles = Chem.MolToSmiles(record.mol)
        self.rows.append({
            "CSD_identifier": record.CSD_identifier,
            "smiles": smiles,
            "resp_charges": json.dumps(charges.tolist()),
        })
        self.charge_arrays[record.CSD_identifier] = charges

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(self.rows)

    def save_csv(self, path: Path) -> None:
        self.to_dataframe().to_csv(path, index=False)
        logger.info("Wrote %d row(s) to %s", len(self.rows), path)

    #zipped numpy library
    def save_npz(self, path: Path) -> None:
        np.savez_compressed(path, **{str(k): v for k, v in self.charge_arrays.items()})
        logger.info("Wrote charge arrays for %d molecule(s) to %s", len(self.charge_arrays), path)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sdf_dir", type=Path, help="Directory containing *.sdf files (one 3D conformer per molecule)")
    parser.add_argument("--working-dir", type=Path, default=Path("psiresp_working_directory"))
    parser.add_argument("--output-csv", type=Path, default=Path("resp_charges.csv"))
    parser.add_argument("--output-npz", type=Path, default=Path("resp_charges.npz"))
    parser.add_argument("--max-iterations", type=int, default=3)
    parser.add_argument("--n-processes", type=int, default=1,
                        help="Number of ESP worker processes (default: 1)")
    parser.add_argument("-v", "--verbose", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    # Psi4 initializes native thread pools. Forking after import can deadlock
    # ESP workers; start clean Python processes instead.
    multiprocessing.set_start_method("spawn", force=True)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    loader = sdfLoader(args.sdf_dir)
    collector = RespResultCollector()

    for record in loader.load():
        if not record.has_conformer:
            continue

        try:
            calc = RespCalculation(record, args.working_dir, max_iterations=args.max_iterations,
                                   n_processes=args.n_processes)
            charges = calc.run_to_completion()
        except Exception:
            logger.exception("[%s] failed, skipping", record.CSD_identifier)
            continue

        collector.add(record, charges)

    collector.save_csv(args.output_csv)
    collector.save_npz(args.output_npz)


if __name__ == "__main__":
    main()
