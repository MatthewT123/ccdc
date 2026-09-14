# CCDC molecular charge experiments

This research project retrieves molecular structures from the Cambridge Structural
Database, computes electrostatic potentials and RESP charges, and experiments with
a molecular variational encoder and a charge-enabled Bayesian flow decoder.
See [AGENTS.md](AGENTS.md) for the code map and known prototype limitations.

The main workflow now runs through three scripts. The original notebooks remain
available and unchanged by this reorganization.

| Stage | Script | Input → output |
| --- | --- | --- |
| 1. Structure retrieval | `scripts/retrieve_structures.py` | CSD identifier CSV or local SDFs → `sdf/`, status CSV, manifest |
| 2. Charge calculation | `scripts/predict_charges.py` | SDFs → all-atom RESP `charges.npz`, CSV, calculation logs |
| 3. Fine-tuning | `scripts/finetune.py` | SDFs + RESP labels + pretrained checkpoint → fine-tuned checkpoints and metrics |

Here “charge prediction” means the quantum-chemistry calculation already implemented
in `esp_generation.py`. Stage 2 reuses that code; it does not run a neural predictor.

## Local setup (Linux x86-64)

Use [pixi](https://pixi.sh/) for the native chemistry dependencies. The ML
environments' PyPI dependencies are installed through pixi's uv integration.
`pixi.lock` records the resolved packages for all environments.

```bash
pixi install --locked
pixi install --locked -e ml
```

The default environment provides Python 3.10, Psi4, PsiRESP, and RDKit. The `ml`
environment provides Python 3.10, CPU PyTorch, PyTorch Geometric, RDKit, and
Open Babel. The optional `ml-gpu` environment adds CUDA support (see below).
The older exported environment and
setup scripts are retained as historical references; use the pixi setup instead.

The chemistry environment pins libint 2.9 because newer libint releases can
resolve successfully but lack the shared-library ABI needed by this Psi4 build.

## Run the three-stage workflow

Run these commands from the repository root. Every `--output` must name a new
directory: scripts refuse to overwrite existing runs. Each script has `--help`.

### 1. Retrieve structures

To use local molecular data immediately, without a CSD installation:

```bash
pixi run retrieve-structures --sdf csd_mol.sdf --output runs/example-structures
```

`--sdf` accepts a single file or a directory containing one molecule per SDF.
It preserves atom order, explicit hydrogens, and components. All inputs need a
parseable 3D conformer. Imported files are written under the output's `sdf/` directory.

For database retrieval, run the same script with the CSV used by `csd_pipeline.ipynb`:

```bash
# Run with a Python environment containing the CCDC API, RDKit, and NumPy:
python scripts/retrieve_structures.py --csv molecules.csv --output runs/csd-structures
```

The required CSV column is `Database identifier`; other columns can remain present.
Identifiers must be unique. As in the notebook, CSD retrieval selects the largest
component by atom count. It does not neutralize, embed, or optimize the structure.
Failures are recorded in `status.csv`, failed files are kept separately for inspection,
and any failed molecule results in a nonzero exit status. The manifest records
successful SDF paths, hashes, and atom identities.

CSD retrieval requires a configured CSD database and licence in addition to the
Python API. The repository's existing pixi environments do not contain the CCDC API.
On a machine with CSD already installed, an optional pixi-managed API environment
can be invoked as follows (this CSD command has not been validated on this machine):

```bash
pixi exec --channel https://conda.ccdc.cam.ac.uk --channel conda-forge \
  --spec 'python=3.11' --spec csd-python-api --spec rdkit --spec numpy \
  python scripts/retrieve_structures.py --csv molecules.csv --output runs/csd-structures
```

This uses the channels documented in the
[CCDC installation notes](https://downloads.ccdc.cam.ac.uk/documentation/API/installation_notes.html).
Installing the API does not supply the database or activate its licence.

#### Install a downloaded API wheel and configure your licence

Obtain the Linux wheel and CSD Portfolio/data installers through the
[CCDC downloads portal](https://www.ccdc.cam.ac.uk/support-and-resources/csdsdownloads/).
Keep private, temporary download URLs out of repository files. Place the wheel in
the ignored `runs/downloads/` directory. To reproduce the separate API environment
used locally (run environment creation only if it does not already exist):

```bash
uv venv --python .pixi/envs/ml/bin/python runs/ccdc-api-env
uv pip install --python runs/ccdc-api-env/bin/python \
  runs/downloads/csd_python_api-3.7.1-py3-none-linux_x86_64.whl rdkit
```

Create a private `.env` in the repository root. These commands preserve an existing
file; edit it locally to replace the placeholder with your activation key:

```bash
cp -n .env.example .env
chmod 600 .env
${EDITOR:-nano} .env
```

The file should contain this setting, with your actual key replacing the placeholder:

```dotenv
CCDC_LICENSING_CONFIGURATION='la-code;YOUR_ACTIVATION_KEY'
```

`.env` and `.env.*` are Git-ignored; only the placeholder `.env.example` is tracked.
Do not put your real key in README, commands committed to Git, or run reports.
The retrieval script loads this setting automatically from the repository-root
`.env` before importing CCDC. An already exported environment variable takes
precedence. The loader reads only this setting; it does not execute shell commands
or expand variables. CCDC also supports `lf-server;URL` for a licence server.
See [CCDC licensing instructions](https://support.ccdc.cam.ac.uk/support/solutions/articles/103000306179-how-do-i-activate-the-software-for-all-users-).
This configures activation on each invocation; it does not install a permanent
system-wide licence. Activation may need network access.

Test API licensing and local molecule reading, without requiring CSD data:

```bash
runs/ccdc-api-env/bin/python - <<'PY'
import sys
sys.path.insert(0, 'scripts')
from _workflow.licensing import load_ccdc_license
load_ccdc_license()
from ccdc import io
with io.MoleculeReader('csd_mol.sdf') as reader:
    molecule = reader[0]
    print(f'API works: {len(molecule.atoms)} atoms, {len(molecule.bonds)} bonds')
PY
```

After installing/configuring the CSD database from the Portfolio installer, retrieve
structures using the same private configuration:

```bash
runs/ccdc-api-env/bin/python scripts/retrieve_structures.py \
  --csv molecules.csv --output runs/csd-structures
```

Licence checking and reading the included SDF succeeded (40 atoms, 41 bonds).
The wheel does not contain the CSD database. A licence error and a missing-database
error require different fixes. If activation fails, check the key's current
entitlement with CCDC.

#### Minimal database installation

The online installer supports selecting just **CSD Main Data**, including its
updates, rather than the full software/data portfolio:

```bash
# Obtain the installer privately from the CCDC portal; keep its URL out of Git.
chmod u+x runs/downloads/CSDInstallerOnline-2026.1.1-linux
QT_QPA_PLATFORM=offscreen runs/downloads/CSDInstallerOnline-2026.1.1-linux search
QT_QPA_PLATFORM=offscreen runs/downloads/CSDInstallerOnline-2026.1.1-linux \
  --root "$PWD/runs/csd-install" --accept-licenses \
  install uk.ac.cam.ccdc.data.csd
```

The inspected catalogue requires approximately **9.95 GB installed plus 3.57 GB
temporary space** for this selection. It includes the maintenance tool and main
data updates, but not Mogul, IsoStar, CrossMiner, or the desktop applications.
These sizes can change with database releases. Keep the installation under ignored
`runs/`; do not commit or redistribute database files.

On Linux the API reads the data location from `~/.config/CCDC/CSD.ini`. Its
`[General]` section should point `root` to the absolute `ccdc-data` directory
inside your installation. Preserve existing settings when changing it. See
[CCDC custom installation instructions](https://support.ccdc.cam.ac.uk/support/solutions/articles/103000306299).

#### Fixed starter train/eval dataset

After database installation, run:

```bash
runs/ccdc-api-env/bin/python scripts/create_csd_dataset.py \
  --output runs/csd-small-v1 --train-size 1000 --eval-size 100 --seed 42
```

This creates `train/sdf/`, `eval/sdf/`, identifier lists `train.csv` and `eval.csv`,
and a manifest recording the database version, selection rules, IDs, and file hashes.
Outputs must be new directories; keep these files fixed for subsequent experiments.
Rebuilding against a different database version can select different molecules.

The starter selection uses single-component molecules with 3–10 heavy atoms,
elements supported by both the model and HF/6-31G* (H, C, N, O, F, P, S, Cl, Br),
no formally charged atoms or radicals, complete 3D data,
no disorder or polymers, no isotopic labels, and crystallographic R factor at most 5%.
RDKit must validate the molecular graph. Heavy-atom separations must be at least
0.65 times the sum of covalent radii, and bonds at most 1.35 times that sum.
Halogens must have valence one; nonstandard stereochemistry is excluded. Saving
and reading the SDF must preserve connectivity and complete explicit hydrogens.
These are screening checks, not a guarantee of chemical or label accuracy.
Iodine is excluded: the neural model supports it, but the current Psi4 6-31G*
basis does not. Basis availability was checked for every retained element.
Stereo-independent canonical SMILES and CSD refcode families are unique across both
splits. This prevents identical connectivity/stereoisomers and repeat crystal
determinations from appearing in both; it is not a scaffold-disjoint benchmark.
The prepared geometry retains crystal heavy-atom coordinates and regenerates all
explicit hydrogen coordinates with RDKit. It does not optimize the geometry.
This differs from the simple largest-component export in `retrieve_structures.py`.

Reserve all 100 evaluation molecules for later testing. Use a validation split
within the 1,000 training molecules for tuning. Structure export does not calculate
RESP labels or start model training. To generate training labels separately:

```bash
pixi run predict-charges --sdf runs/csd-small-v1/train/sdf \
  --output runs/csd-small-v1-train-charges
```

The local `runs/csd-small-v1` export contains **1,000 train + 100 eval molecules**,
all verified by reading their saved SDFs and checking hashes, atom counts, hydrogen
completeness, and unique connectivity/refcode families. `verification.json` records
the audit. There are 9 molecules with 3 heavy atoms, 16 with 4, 37 with 5, 61 with 6,
90 with 7, 189 with 8, 248 with 9, and 450 with 10.
Earlier drafts are preserved separately; use only the final `csd-small-v1` dataset.
The final run used seed 42 and the verified candidate list
`runs/csd-small-basis-candidates.csv` via `--candidates-csv`; its hash is in the
manifest. This option reapplies all filters to a fixed candidate list.
A real Psi4/RESP smoke calculation passed on the selected small molecule QOBGUL03,
with finite, atom-aligned charges summing to its formal charge. The remaining
dataset still needs reference-charge calculation before training.

### 2. Calculate reference charges

```bash
pixi run predict-charges \
  --sdf runs/example-structures/sdf \
  --output runs/example-charges
```

For CSD structures, substitute `runs/csd-structures/sdf`. Useful options include
`--n-processes 1`, `--max-iterations 3`, and `--verbose`. Psi4/RESP runs on CPU.
The calculations can be expensive for larger molecules.

Outputs are `charges.npz` (identifier → all-atom charge array), `charges.csv`
(`CSD_identifier`, `smiles`, `charges`, `atom_indices`), `manifest.json`, `status.csv`,
and quantum work/log files. Each charge vector must be finite, match the source atom
count, and sum to the molecule's formal charge. A failure produces a nonzero exit
status while preserving successful results. Inspect `status.csv` before training;
fine-tuning rejects structures with missing labels. PsiRESP defaults are preserved,
including the default single-point workflow without geometry optimization.

### 3. Fine-tune the pretrained model

Download the official checkpoint using the section below, then run:

```bash
# One-molecule execution check with the included example; no held-out validation:
pixi run -e ml-gpu finetune \
  --sdf runs/example-structures/sdf \
  --charges runs/example-charges/charges.npz \
  --output runs/example-finetune \
  --device cuda:0 --epochs 2 --batch-size 1 --val-fraction 0
```

For a larger dataset, use a held-out validation split:

```bash
pixi run -e ml-gpu finetune \
  --sdf runs/csd-structures/sdf --charges runs/csd-charges/charges.npz \
  --output runs/csd-finetune --device auto \
  --epochs 10 --batch-size 8 --lr 0.0001 --val-fraction 0.2
```

The second example assumes stage 2 was run on the CSD structures with output
`runs/csd-charges`. For CPU use environment `ml` and `--device cpu`.
Defaults are 10 epochs, batch size 8, constant learning rate `1e-4`, 20% validation,
seed 42, two CPU threads, and gradient clipping at 1.0. CLI arguments override
these run settings; `--config` selects the model YAML and `--checkpoint` selects
the initial weights. Device configuration is described further below.

The script loads the pretrained encoder, latent projections, and structural decoder;
only the new charge head is randomly initialized. Loading rejects unexpected or
missing backbone weights rather than silently training from scratch. By default
all model parameters are fine-tuned. `--trainable head` freezes everything except
the new charge head. The training objective is the existing notebook's
`TrainLoopCharges.training_step`: reconstruction, charge, and KL losses. W&B
uploads are disabled by default; enable them with `--wandb online` (or save locally
with `--wandb offline`). This script bypasses unfinished notebook dataset and
validation plumbing without modifying those notebooks.

Training uses heavy atoms with bonded hydrogen charges absorbed into them, preserving
the molecular charge sum. Supported heavy atoms are C, N, O, F, P, S, Cl, Br, I.
Labels must be indexed by SDF filename stem and contain one charge for every source
atom, including explicit H. The stage-2 manifest verifies the exact source SDF hash
and atom identities. Legacy NPZ files such as `resp_charges.npz` can also be used,
but only their counts, finite values, and charge sums can be checked; the script
reports that their original atom ordering cannot be verified from a manifest.

`--val-fraction` splits by molecule identifier using the seed; validation molecules
are excluded from training. At least two molecules are required unless the fraction
is zero. Validation MAE/RMSE compare absorbed heavy-atom reference charges with a
deterministic decoder evaluation at time 1 on the supplied geometry. They do not
measure molecule-generation quality. A tiny execution check does not establish
useful predictive accuracy.

The output directory contains:

- `best.ckpt`: lowest validation MAE, or lowest training loss if validation is disabled.
- `last.ckpt`: final epoch, including weights, config, optimizer state, and provenance.
- `metrics.jsonl`: per-epoch loss components, charge diagnostics, timing, and CUDA memory.
- `steps.jsonl`: optimizer-step losses, learning rate, and gradient norm.
- `before.json`, `after.json`: fixed-seed diagnostics before and after fine-tuning.
- `wandb.json`: run URL and ID when tracking is enabled.
- `run.json`, `config.json`, `result.json`: data/checkpoint hashes, split IDs, settings,
  weight-loading report, and completion status.

Passing a fine-tuned checkpoint back via `--checkpoint` starts another fine-tuning
run with a new optimizer; it is not an exact optimizer/RNG resume operation.

### Track reconstruction and charges with W&B

Authenticate with `pixi run -e ml-gpu wandb login`, then add `--wandb online`.
Use `--wandb-project` and `--wandb-entity` to choose the destination; never put an
API key in source code. `--wandb-run-id` resumes tracking only, not training state.
Metrics are saved locally even when W&B is disabled.

For the local pilot, existing validated Psi4/RESP labels were reused for 200
molecules from the original training pool. The smallest 200 successfully labelled
molecules were shuffled with seed 42 into 100 train and 100 test molecules;
`runs/csd-pilot-100x100/selection.json` records the selection. The original fixed
100-molecule evaluation set remains untouched. This pilot is selected from
successful calculations and is not a representative benchmark of the whole CSD.

```bash
pixi run -e ml-gpu finetune \
  --sdf runs/csd-pilot-100x100/train/sdf \
  --charges runs/csd-pilot-100x100/train/charges/charges.npz \
  --test-sdf runs/csd-pilot-100x100/test/sdf \
  --test-charges runs/csd-pilot-100x100/test/charges/charges.npz \
  --output runs/pilot-next-run --device cuda:0 \
  --epochs 1 --batch-size 8 --val-fraction 0 \
  --reconstruction-molecules 100 --sample-steps 100 --wandb online
```

Use a fresh output directory. Separate test inputs are optional, must have reference
labels, and are rejected if they overlap training by connectivity or refcode family.
Test data never contributes gradients or checkpoint selection. No Psi4 calculation
is launched by fine-tuning; it consumes existing labels.

W&B `train_*` metrics show optimizer-step losses. `before_*` and `epoch_diagnostics_*`
record training/test diagnostics; final `after_*` values appear in the run summary.
`evaluation_*` metrics share an `evaluation_epoch` axis for before/after curves.
`epoch_performance_*` reports optimizer-epoch seconds, molecules/second, and peak
PyTorch allocated/reserved GiB. These memory figures exclude other applications and
CUDA allocations outside PyTorch; epoch time excludes before/after diagnostics.

- `given_geometry_charges`: deterministic charge predictions at decoder time 1
  with supplied atom types and positions. MAE/RMSE are in elementary-charge units
  (`e`), using heavy-atom charges with bonded H charges absorbed.
- `latent_only` / `test_latent_only`: decode from latent representations with the
  atom count supplied, without supplying original types or coordinates to the
  decoder. A position-based Hungarian assignment in the centered encoder frame
  gives atom-type accuracy, position RMSE in angstroms, and matched charge errors.
  This does not measure bond accuracy or prediction of the atom count.
- `charge_loss` is the weighted sum of molecule-averaged atomic MSE and molecular
  total-charge MSE. It is not itself charge RMSE. For plain atomic MSE, 0.01, 0.0025,
  and 0.0004 e² correspond to RMSE 0.10, 0.05, and 0.02 e respectively. Compare with
  the logged zero-charge baseline; an acceptable error depends on the downstream use.
- Coordinate/type losses retain the original stochastic BFN objective. The existing
  position KL expression omits a fixed prior-normalization constant, so the logged
  KL component can be negative; its absolute value is not a reconstruction score.

For future label generation, `scripts/batch_resp.py --sdf ... --output ...` runs
four independent molecule jobs by default (`--workers` changes this). `--resume`
reuses validated completed attempts in that output directory. `--precise-fit`
uses a dense least-squares RESP solve for small systems, checks linear residuals,
and rejects unconverged fits; it retains the original fitting objective. This
avoids inaccurate solutions from PsiRESP's default singular-matrix fallback.
Inspect `status.csv`: failures are preserved for diagnosis, and a nonzero exit
means the label set is incomplete. In particular, the default PsiRESP ESP-grid
radii do not support Br; those failed jobs are excluded from this cached pilot.

The completed 100/100 pilot exposed a label-quality problem: absorbed reference
charges reach 39.29 e in training and 37.84 e in testing, while the charge head is
bounded to ±2 e. Hash alignment, convergence, and molecular charge conservation
passed, but those checks do not establish scientifically usable labels. Preserve
these results for diagnosis; audit the RESP calculation/fitting pipeline before
using this label set for further accuracy experiments. No labels were clipped or
removed after inspecting test results.

### Verification

The three stages were exercised with two local molecules and real Psi4/RESP charges.
Two epochs of full-model fine-tuning passed on CPU; head-only fine-tuning passed on
the RTX 5090. Saved weights confirmed that full-model tuning changed pretrained
parameters and head-only tuning preserved them. The separate CCDC environment now
passes API licensing and live database access using the ignored `.env` and minimal
CSD Main Data installation (1,451,367 entries). Retrieval of HXACAN06 passed.
The original CSV exported 11 of 13 entries: ALESOC failed RDKit valence validation
and FAFYIZ was absent from the database. The CSD adapter also has a simulated
contract test. All notebooks were left byte-for-byte unchanged.

```bash
pixi run -e ml python -m unittest discover -s tests
```

## Download the full pretrained MolFLAE checkpoint

The [official MolFLAE instructions](https://github.com/MuZhao2333/MolFLAE/tree/master/Latent_Experiments)
link to the pretrained ZINC-9M model on
[Google Drive](https://drive.google.com/file/d/161pBWbsbkZbN4r57XsuWU6QzYA5nuvAB/view).
From this repository's root, download it with:

```bash
mkdir -p MolFLAE/ckpt-zinc9M
wget --no-clobber --timeout=30 --tries=3 \
  -O 'MolFLAE/ckpt-zinc9M/model-epoch=24-val_loss=3.40.ckpt' \
  'https://drive.usercontent.google.com/download?id=161pBWbsbkZbN4r57XsuWU6QzYA5nuvAB&export=download&confirm=t'
echo '6eb8332e9f2c955fe93197fca54213c2d561e5078067a6c3102f11ce7b9a7f3f  MolFLAE/ckpt-zinc9M/model-epoch=24-val_loss=3.40.ckpt' | sha256sum --check
```

`--no-clobber` preserves an existing destination. If an interrupted download or a
Drive error page fails the checksum, move that file aside before retrying. The
checksum above was calculated from the verified download on 2026-09-14; it is not
an independently published author checksum. The verified file is 66,892,474 bytes
(about 64 MiB). Downloaded checkpoints are ignored by Git.

This is a Lightning checkpoint from epoch 24 / global step 145672. Its
`state_dict` includes the encoder, latent projections, and structural decoder;
it also contains optimizer/training state. Inspect it without loading arbitrary
pickled classes using the ML environment:

```bash
pixi run -e ml python -c "import torch; c = torch.load('MolFLAE/ckpt-zinc9M/model-epoch=24-val_loss=3.40.ckpt', map_location='cpu', weights_only=True); print(c['epoch'], sorted({k.split('.')[0] for k in c['state_dict']}))"
```

All four files already tracked in `MolFLAE/weights/` exactly match their
corresponding tensors in this official checkpoint. They provide only the encoder
and latent projections; the full download also provides decoder weights.
The official checkpoint does not contain this project's added `charge_head`.
Downloading it does not change the old smoke checks: their training step still
initializes a new model. The new `finetune.py` performs checked pretrained loading
and has verified compatibility with this checkpoint.

## Run small checks

From the repository root:

```bash
pixi run resp-help
pixi run smoke-resp
pixi run -e ml smoke-ml
```

- `smoke-resp` generates a water SDF, runs the actual RESP command-line pipeline,
  and checks that CSV and NPZ contain matching finite charges with zero net charge.
- `smoke-ml` loads the checked-in encoder weights, encodes `csd_mol.sdf`, and runs
  one optimizer update through `TrainLoopCharges` using `resp_charges.npz`.
  Hydrogen charges are transferred to bonded heavy atoms before hydrogen removal.
  It checks finite latent values, loss, charge-head gradients, and parameter updates.
  W&B is disabled. The charge decoder starts untrained; this is a runtime check,
  not a trained predictor or an accuracy benchmark.

Each invocation writes to a new directory under ignored `runs/`, including a
`result.json` on success. The ML run also writes `latent.npz`; the RESP run saves
its CLI log, quantum calculation files, and charge exports. Existing research
artifacts are not overwritten. The RESP smoke script has a five-minute timeout.

To process your own SDF directory, choose fresh output paths:

```bash
pixi run python esp_generation.py /path/to/sdf_files \
  --working-dir runs/my-resp-work \
  --output-csv runs/my-resp-charges.csv \
  --output-npz runs/my-resp-charges.npz \
  --max-iterations 3 --verbose
```

The CLI logs and skips failed molecules; check the output records as well as its
exit status. This workflow reads one molecule from each SDF and preserves explicit
hydrogens. Calculations on larger molecules may take much longer than the water check.
The CLI uses fresh (`spawn`) ESP worker processes to avoid a native-library fork
hang and defaults to one worker; set `--n-processes` to control ESP parallelism.
PsiRESP's defaults did a single-point calculation in the water check, without
geometry optimization.

Both smoke checks passed locally on 2026-09-14. The example molecule yielded
22 heavy atoms, latent shapes `(10, 32)` and `(10, 3)`, and a finite training loss
of about 35.47 for the seeded update. The water charges were approximately
`[-0.79462, 0.39771, 0.39691]`, summing to zero within floating-point precision.

## Run MolFLAE on an NVIDIA GPU

```bash
pixi install --locked -e ml-gpu
pixi run -e ml-gpu smoke-gpu
```

This environment uses PyTorch 2.8 with CUDA 12.8 and matching PyTorch Geometric
extensions. CUDA 12.8 builds support Blackwell cards such as the RTX 5090;
see the [PyTorch Blackwell support announcement](https://pytorch.org/blog/pytorch-2-7/).
An operational NVIDIA driver is required. The existing `ml` environment remains
CPU-only; simply selecting `cuda` there will not enable GPU execution.

`smoke-gpu` passes `--device cuda`, moves the encoder, training model, and batch
onto the GPU, and checks that latents, loss, and gradients actually reside there.
It fails if CUDA is unavailable. Its `result.json` includes GPU name, CUDA version,
and peak allocated GPU memory. This remains a one-molecule runtime check rather
than a throughput benchmark or full training run. The Psi4/RESP task still uses CPU.
Verified locally on the RTX 5090: encoder forward pass and charge-head optimizer
update both passed on CUDA, with about 247 MB peak allocated GPU memory.

For GPU notebooks, select `.pixi/envs/ml-gpu/bin/python`. In the training notebook,
device selection now follows the shared configuration below. Batches must be moved
to the model's device before calling its training methods directly:

```python
train_loop = TrainLoopCharges(config)
train_loop.configure_optimizers()
batch = batch.to(train_loop.device)
```

## Configure the compute device

Scripts and notebooks use this selection order:

1. Explicit `--device` argument (or `device=` in the Python API).
2. `MOLFLAE_DEVICE` environment variable.
3. `runtime.device` in `MolFLAE/config.yaml`.
4. `auto`: the current CUDA device when available, otherwise CPU.

The checked-in config uses `auto`. Supported values are `auto`, `cpu`, `cuda`, and
an indexed GPU such as `cuda:0` (indices follow `CUDA_VISIBLE_DEVICES`). Explicitly
requesting unavailable CUDA raises an error instead of silently using CPU.

```bash
pixi run -e ml-gpu smoke-ml --device cuda:0
pixi run -e ml-gpu smoke-ml --device cpu
MOLFLAE_DEVICE=cuda:0 pixi run -e ml-gpu smoke-ml
```

For a persistent project default, edit:

```yaml
runtime:
  device: auto
```

In Python, `TrainLoop(config, device=...)` and `TrainLoopCharges(config, device=...)`
place the whole model on the selected device. For an independently constructed
encoder, use `encoder.to(resolve_device(config=config))` with
`from utils.device import resolve_device`. Both decoder classes accept an optional
`device=` argument and follow subsequent `.to(...)` calls. Schedule tensors move
with the model; they do not add new keys to existing checkpoints. Create the
optimizer after choosing model placement. Notebook dataset preprocessing stays on
the host; batches move to the configured device for computation. `.cpu()` calls
used to export NumPy/JSON data or collect logs are intentional host transfers.

The `smoke-gpu` task is a convenience alias selecting CUDA explicitly. To test
configuration and device moves, including a short charge-decoder sampling run:

```bash
pixi run -e ml python -m unittest discover -s tests -p test_devices.py
pixi run -e ml-gpu python -m unittest discover -s tests -p test_devices.py
```

## Notebooks and CSD access

All environments include `ipykernel`. In your notebook editor, select
`.pixi/envs/default/bin/python` for chemistry or `.pixi/envs/ml/bin/python` for ML.
Use the repository root as the working directory for root notebooks and
`MolFLAE/` for its notebooks. Some notebook cells remain unfinished or depend on
historical paths; a passing smoke check does not establish that Run All works.

The CSD retrieval notebooks additionally require the CCDC Python API and an
accessible CSD installation/licence. These are not bundled in the pixi environments.
The included SDF lets the chemistry and ML checks run without CSD access.
