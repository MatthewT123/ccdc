# CCDC molecular charge experiments

This research project retrieves molecular structures from the Cambridge Structural
Database, computes electrostatic potentials and RESP charges, and experiments with
a molecular variational encoder and a charge-enabled Bayesian flow decoder.
See [AGENTS.md](AGENTS.md) for the code map and known prototype limitations.

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
move both model and batches to CUDA and set the decoder's device before construction:

```python
config["decoder_config_charge"]["device"] = "cuda"
train_loop = TrainLoopCharges(config).to("cuda")
train_loop.configure_optimizers()
batch = batch.to("cuda")
```

Selecting the GPU kernel alone does not move model tensors or data to CUDA.

## Notebooks and CSD access

All environments include `ipykernel`. In your notebook editor, select
`.pixi/envs/default/bin/python` for chemistry or `.pixi/envs/ml/bin/python` for ML.
Use the repository root as the working directory for root notebooks and
`MolFLAE/` for its notebooks. Some notebook cells remain unfinished or depend on
historical paths; a passing smoke check does not establish that Run All works.

The CSD retrieval notebooks additionally require the CCDC Python API and an
accessible CSD installation/licence. These are not bundled in the pixi environments.
The included SDF lets the chemistry and ML checks run without CSD access.
