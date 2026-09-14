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
uploads are disabled. This script bypasses unfinished notebook dataset and
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
- `metrics.jsonl`: per-epoch training loss and validation charge MAE/RMSE.
- `run.json`, `config.json`, `result.json`: data/checkpoint hashes, split IDs, settings,
  weight-loading report, and completion status.

Passing a fine-tuned checkpoint back via `--checkpoint` starts another fine-tuning
run with a new optimizer; it is not an exact optimizer/RNG resume operation.

### Verification

The three stages were exercised with two local molecules and real Psi4/RESP charges.
Two epochs of full-model fine-tuning passed on CPU; head-only fine-tuning passed on
the RTX 5090. Saved weights confirmed that full-model tuning changed pretrained
parameters and head-only tuning preserved them. CSD database access is unavailable
locally, so only the CSD adapter contract and its missing-installation error were
tested, not live database retrieval. All notebooks were left byte-for-byte unchanged.

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
