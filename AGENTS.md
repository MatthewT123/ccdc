# Working instructions

- Avoid irreversible changes, especially to untracked files; confirm with the user first if needed.
- Preserve unrelated work and inspect `git status` before editing. Commit regularly when easy; stage only intended files.
- Treat checked-in SDF/CSV/NPZ data and PTH model weights as research artifacts. Do not overwrite or regenerate them incidentally.
- Keep notebook edits focused; avoid changing execution counts, outputs, or metadata unnecessarily.

# Project purpose

This is a Python/Jupyter computational chemistry research prototype. It retrieves molecular structures from the Cambridge Structural Database (CSD), computes electrostatic potentials (ESP) and RESP atomic charges, and experiments with a MolFLAE variational molecular encoder and Bayesian flow decoder extended to predict charges.

These are partially connected research workflows, not a packaged application or a verified end-to-end training pipeline. Distinguish implemented code, notebook experiments, and validated results when answering questions.

# Repository map

| Path | Purpose |
| --- | --- |
| `molecules.csv` | Input compound list; CSD lookup key is `Database identifier`. |
| `csd_pipeline.ipynb` | Reads the CSV, retrieves CSD entries, selects the largest component, writes `sdf_files/<identifier>.sdf`, and records success/failure in `molecules_with_status.csv`. |
| `csd_test.ipynb` | Exploratory CSD retrieval, minimization, conformer generation, and RDKit neutralization/embedding. Produces the example `csd_mol.sdf`. |
| `esp_generation.py` | Main command-line workflow: load SDFs, drive PsiRESP/Psi4 jobs, collect RESP charges in CSV and compressed NPZ. Despite its name, its exported result is atomic charges, not an ESP grid. |
| `psi4_test.ipynb` | Direct Psi4 ESP grid calculation, PyVista visualization, and manual PsiRESP/CLI output comparisons. |
| `utils/molecule_processing.py` | RDKit `neutralize_atoms`; modifies the supplied molecule in place. |
| `utils/visualize.py` | CCDC/RDKit molecular and conformer visualization using py3Dmol. |
| `MolFLAE/encoder_test.ipynb` | Loads encoder/latent-layer weights, extracts latent representations, and experiments with charge-head gradients and rotation invariance. |
| `MolFLAE/train_model.ipynb` | Experimental charge training, heavy-atom charge preparation, and unfinished `RDKitChargeDataset`. |
| `MolFLAE/model/encoder.py` | Molecular graph encoder producing global-node features and positions. |
| `MolFLAE/model/encoder_standalone_cpu.py` | Standalone encoder and `molecule_to_latent` helper. |
| `MolFLAE/model/uni_transformer.py` | Attention layers coupling node features and 3D coordinates. |
| `MolFLAE/model/bfn4sbdd.py` | Original `BFN4SBDDScoreModel` and charge-enabled `BFN_charge`: coordinate/type reconstruction and scalar atomic-charge prediction. |
| `MolFLAE/model/train_loop.py` | Lightning wrappers `TrainLoop` and `TrainLoopCharges`, KL/reconstruction losses, validation, and sampling. |
| `MolFLAE/utils/data_loading.py` | Recursive SDF loader returning atom-type indices, positions, and atom counts; does not load charge labels. |
| `MolFLAE/config.yaml` | Encoder, decoder, charge losses, training, and evaluation settings. |
| `MolFLAE/weights/` | Encoder and latent projection weights explicitly loaded by `encoder_test.ipynb`. |

# Data contracts and scientific assumptions

- The bulk CSD export selects the largest component but does not apply the neutralization/minimization steps from `csd_test.ipynb`. Do not describe these as one identical preparation pipeline.
- `esp_generation.py` scans only immediate `*.sdf` children, reads the first molecule from each file, preserves explicit hydrogens (`removeHs=False`), and derives its identifier from the filename stem. It skips records without conformers.
- RESP CSV columns are `CSD_identifier`, `smiles`, and `resp_charges` (a JSON array). The NPZ maps identifiers to charge arrays. Maintain atom order when pairing charges with coordinates.
- ESP grid output in `psi4_test.ipynb` is `esp_data.npz`, with `esp`, `origin`, `spacing`, `shape`, and `points`. Coordinate units and array flattening order need careful verification for scientific changes.
- The model's supported heavy atoms are C, N, O, F, P, S, Cl, Br, I: atomic numbers `[6, 7, 8, 9, 15, 16, 17, 35, 53]` map to indices `0..8`.
- `process_sdf_files_to_list` returns indexed `h` values. `TrainLoopCharges.training_step` instead expects atomic numbers in `h` (normally `[N, 1]`), `x` coordinates `[N, 3]`, `charges` `[N, 1]`, and graph membership `batch` `[N]`. Do not connect these interfaces without explicit conversion.
- `molecule_to_latent` guesses whether `h` contains atomic numbers using `h.max() >= K`; this is ambiguous for molecules containing only low atomic numbers such as C/N/O. Prefer explicit indexed input to this helper.
- The training notebook explores transferring each explicit hydrogen's charge to its bonded heavy atom before removing H. Earlier cells simply drop H. Preserve the intended total-charge convention explicitly.
- `BFN_charge` predicts per-atom charges bounded by `2 * tanh(...)`; its loss includes per-atom error and a soft molecular total-charge penalty. This does not establish model accuracy or exact charge conservation.
- `TrainLoopCharges` initializes new encoder/decoder modules; it does not automatically load the checked-in encoder weights.

# Environments and execution

- Root CSD notebooks document a separate `ccdc-env` installed from the CCDC Conda channel. They require the CCDC API and access to a local CSD installation.
- `psi4_environment.yml` is a detailed Linux/Python 3.9 Conda export. The notebook documents `conda env create -n psiresp-new -f psi4_environment.yml`.
- For the verified local setup, use `pixi.toml` and `pixi.lock` instead of the historical setup scripts. The user requests uv for Python packages and pixi for Conda packages; pixi uses uv for the ML environment's PyPI dependencies. See `README.md` for commands.
- `pixi install --locked` installs the default chemistry environment; `pixi install --locked -e ml` installs CPU ML. Both use Python 3.10. Psi4 1.9.1 requires the pinned libint 2.9 ABI; allowing libint 2.13 resolved but failed to import in the local setup.
- ML environment files are in `MolFLAE/molflae-env-setup/`, targeting `MolFLAE2` and Python 3.10. Review scripts and dependency compatibility before using them: `setup.sh` currently lacks a pip-install command before its PyG extension list.
- Run root notebooks with the repository root as cwd; run MolFLAE notebooks with `MolFLAE/` as cwd. Imports such as `model.*`, `utils.*`, and relative `config.yaml`/weight paths depend on this. Root and MolFLAE contain different `utils` directories.
- Notebook paths referencing `/root/ccdc`, Windows scratch folders, or absent `data/latent_experiment/val` and `ckpt-zinc9M` directories are examples, not portable defaults.
- Do not launch expensive quantum calculations, full training, environment installation, or W&B logging merely to summarize or inspect the project.

Example RESP invocation from the repository root, after activating a suitable environment (not verified by this documentation review):

```bash
python esp_generation.py /path/to/sdf_files \
  --working-dir /tmp/ccdc-resp-work \
  --output-csv /tmp/ccdc-resp-charges.csv \
  --output-npz /tmp/ccdc-resp-charges.npz \
  --max-iterations 3 --verbose
```

Use fresh output paths. A two-stage optimization/single-point job can need three `job.run()` calls; the CLI now defaults to three. The tested PsiRESP defaults did only a single-point calculation on water, so do not assume geometry optimization is enabled. The runner executes generated `*/run_*.sh` files and writes adjacent logs. Per-molecule exceptions are logged and skipped, so a successful process exit alone does not prove all molecules succeeded.

The CLI now selects multiprocessing `spawn` and defaults to one ESP worker (`--n-processes`). The previous fork-based execution stalled waiting for an ESP worker after Psi4 initialization, both inside and outside the managed sandbox. Notebook/library callers must handle their multiprocessing context separately.

## Verified local smoke runs (2026-09-14)

- `pixi run smoke-resp`: real water SDF -> Psi4 single-point -> ESP -> RESP CSV/NPZ, with matching finite charges and approximately zero total charge. This verifies runtime behavior, not chemical accuracy across the dataset.
- `pixi run -e ml smoke-ml`: checked-in encoder/latent-layer weights load; the example molecule has 22 heavy atoms and produces `Zh` shape `[10, 32]` and `Zx` shape `[10, 3]`. One actual `TrainLoopCharges` optimizer update has finite loss/gradients and changes charge-head parameters. W&B is disabled; the decoder is newly initialized.
- Scripts create unique output directories under ignored `runs/`; environment files live under ignored `.pixi/`. Smoke success writes `result.json`. Do not commit generated outputs.
- No local CSD installation was found in the usual locations; database retrieval and notebook Run All remain unverified. CCDC API/data/licence are not included in these environments.

# Known prototype gaps and validation

Verified by source inspection; recheck before relying on these observations after changes:

- `RDKitChargeDataset` stores `self.charge_arrays` but reads `self.charges` in `get()`, and emits `charge_arrays` rather than the training wrapper's required `charges` field.
- The original `TrainLoop` references `BFN4SBDDScoreModel` without importing it.
- `encoder_test.ipynb` imports `utils.testing.random_rotation`, but no such module is checked in.
- `config.yaml` contains `max_grad_norm: Q` and large machine-specific GPU/worker settings; it is not a verified ready-to-run configuration.
- Some notebook cells depend on missing or out-of-order variables. Saved outputs are historical evidence, not proof that Run All succeeds today.
- There is no broad automated test suite or CI configuration. `scripts/smoke_resp.py` and `scripts/smoke_ml.py` provide focused executable checks; `README.md` and the pixi manifest/lock document the local setup.
- For documentation-only changes, inspect the diff and run `git diff --check`. For code changes, use focused checks in the correct scientific environment; meaningful checks include atom/charge alignment, total charge after hydrogen absorption, charge-head gradients, and rotation behavior. Clearly report which checks actually ran.
