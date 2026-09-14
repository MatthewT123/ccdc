"""Fine-tune pretrained MolFLAE on atom-aligned RESP labels, without notebooks."""

import argparse
import copy
import json
import math
import os
from pathlib import Path
import sys

from _workflow.data import digest, load_labels, new_output, structures, write_json

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CHECKPOINT = ROOT / "MolFLAE/ckpt-zinc9M/model-epoch=24-val_loss=3.40.ckpt"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sdf", type=Path, required=True)
    parser.add_argument("--charges", type=Path, required=True, help="All-atom reference charges.npz keyed by SDF filename stem")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--config", type=Path, default=ROOT / "MolFLAE/config.yaml")
    parser.add_argument("--output", type=Path, required=True, help="New output directory")
    parser.add_argument("--device", default=None, help="auto, cpu, cuda, cuda:N; overrides env/config")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--trainable", choices=["all", "head"], default="all")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    args = parser.parse_args(argv)
    if min(args.epochs, args.batch_size, args.threads) < 1 or not math.isfinite(args.lr) or args.lr <= 0:
        parser.error("Epochs, batch size, threads, and learning rate must be positive")
    if not 0 <= args.val_fraction < 1 or not math.isfinite(args.max_grad_norm) or args.max_grad_norm <= 0:
        parser.error("Require 0 <= val-fraction < 1 and positive max-grad-norm")
    if not args.checkpoint.is_file():
        parser.error(f"Pretrained checkpoint not found: {args.checkpoint}; see README download instructions")
    os.environ["WANDB_MODE"] = "disabled"
    from _workflow.training import load_config, resolve_device, load_pretrained, molecular_data, validation_metrics
    import torch
    from torch_geometric.loader import DataLoader
    import wandb

    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    records = structures(args.sdf)
    labels, verified = load_labels(args.charges, records)
    data = molecular_data(records, labels)
    if len(data) < 2 and args.val_fraction > 0:
        parser.error("At least two molecules are needed for a validation split; use --val-fraction 0 for a one-molecule smoke run")
    indices = torch.randperm(len(data), generator=torch.Generator().manual_seed(args.seed)).tolist()
    n_val = max(1, int(len(data) * args.val_fraction)) if args.val_fraction else 0
    val_indices, train_indices = indices[:n_val], indices[n_val:]
    config = copy.deepcopy(load_config(args.config))
    device = resolve_device(args.device, config)
    output = new_output(args.output)
    config["evaluation"]["save_dir"] = str(output / "samples")
    config["runtime"] = {"device": str(device)}
    config["train"]["batch_size"] = args.batch_size
    config["train"]["optimizer"]["lr"] = args.lr
    model, loaded = load_pretrained(config, args.checkpoint, device)
    if args.trainable == "head":
        for name, parameter in model.named_parameters():
            parameter.requires_grad_(name.startswith("decoder.charge_head."))
    model.configure_optimizers()
    train_loader = DataLoader([data[i] for i in train_indices], batch_size=args.batch_size, shuffle=True,
                             generator=torch.Generator().manual_seed(args.seed), num_workers=0)
    val_loader = DataLoader([data[i] for i in val_indices], batch_size=args.batch_size, num_workers=0)
    info = {"format": "ccdc-finetune-v1", "device": str(device), "seed": args.seed,
        "torch_version": str(torch.__version__), "batch_size": args.batch_size,
        "val_fraction": args.val_fraction, "max_grad_norm": args.max_grad_norm,
        "checkpoint": str(args.checkpoint.resolve()), "checkpoint_sha256": digest(args.checkpoint), **loaded,
        "charges": str(args.charges.resolve()), "charges_sha256": digest(args.charges),
        "labels_provenance_verified": verified, "charge_convention": "heavy_atom_absorb_h",
        "trainable": args.trainable, "epochs": args.epochs, "lr": args.lr,
        "train_ids": [records[i][0] for i in train_indices], "validation_ids": [records[i][0] for i in val_indices],
        "source_sha256": {identifier: digest(path) for identifier, path, _ in records},
        "selection_metric": "validation_mae" if n_val else "training_loss"}
    write_json(output / "run.json", info)
    write_json(output / "config.json", config)
    if not verified:
        print("Legacy labels: atom counts checked; no manifest available to verify source atom ordering.")
    best = float("inf")
    initial_head = {k: v.detach().clone() for k, v in model.decoder.charge_head.state_dict().items()}
    step = 0
    with wandb.init(mode="disabled"):
        with (output / "metrics.jsonl").open("w") as metrics_file:
            for epoch in range(1, args.epochs + 1):
                model.train()
                total, graphs = 0.0, 0
                for batch in train_loader:
                    batch = batch.to(device)
                    model.optim.zero_grad(set_to_none=True)
                    loss = model.training_step(batch, step)
                    if loss is None or not torch.isfinite(loss):
                        raise ValueError(f"Non-finite loss at step {step}")
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],
                        args.max_grad_norm, error_if_nonfinite=True)
                    model.optim.step()
                    total += loss.item() * batch.num_graphs
                    graphs += batch.num_graphs
                    step += 1
                validation = validation_metrics(model, val_loader, device)
                metrics = {"epoch": epoch, "step": step, "training_loss": total / graphs, "validation": validation}
                metrics_file.write(json.dumps(metrics, allow_nan=False) + "\n")
                metrics_file.flush()
                print(json.dumps(metrics, allow_nan=False), flush=True)
                score = validation["mae"] if validation else total / graphs
                payload = {"format": "ccdc-finetune-v1", "state_dict": model.state_dict(), "config": config,
                    "epoch": epoch, "global_step": step, "optimizer_state": model.optim.state_dict(),
                    "charge_convention": "heavy_atom_absorb_h", "metrics": metrics, "source": info}
                torch.save(payload, output / "last.ckpt")
                if score < best:
                    best = score
                    torch.save(payload, output / "best.ckpt")
                model.train_losses.clear()
    changed = any(not torch.equal(v, model.decoder.charge_head.state_dict()[k]) for k, v in initial_head.items())
    if not changed:
        raise ValueError("No charge-head parameters changed during training")
    write_json(output / "result.json", {"steps": step, "charge_head_updated": changed,
        "best_score": best, "selection_metric": info["selection_metric"], "output": str(output)})
    print(f"Fine-tuning complete. Checkpoints and metrics: {output}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (ValueError, OSError, ImportError, RuntimeError) as exc:
        sys.exit(str(exc))
