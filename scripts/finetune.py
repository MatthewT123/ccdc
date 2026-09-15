"""Fine-tune pretrained MolFLAE on atom-aligned RESP labels, without notebooks."""

import argparse
import copy
import json
import math
import os
from pathlib import Path
import sys
import time
from collections import defaultdict

from _workflow.data import digest, load_labels, new_output, structures, write_json

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CHECKPOINT = ROOT / "MolFLAE/ckpt-zinc9M/model-epoch=24-val_loss=3.40.ckpt"


def wandb_diagnostic_metrics(result, prefix='evaluation'):
    """Select only reconstruction and charge losses for W&B.

    Full diagnostics remain in before.json/after.json and metrics.jsonl. W&B is
    deliberately limited to the two requested losses for each available split.
    """
    selected = {}
    for dataset, payload in result.items():
        objectives = payload.get('objectives', {})
        for source, name in (('structure_loss', 'reconstruction_loss'),
                             ('charge_loss', 'charge_prediction_loss')):
            value = objectives.get(source)
            if isinstance(value, (int, float)) and math.isfinite(value):
                selected[f'{prefix}/{dataset}/{name}'] = value
    return selected


def wandb_training_metrics(values, prefix='train'):
    """Select the two optimization losses from a per-step or epoch mapping."""
    selected = {}
    for source, name in (('structure_loss', 'reconstruction_loss'),
                         ('charge_loss', 'charge_prediction_loss')):
        value = values.get(source)
        if isinstance(value, (int, float)) and math.isfinite(value):
            selected[f'{prefix}/{name}'] = value
    return selected


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sdf", type=Path, required=True)
    parser.add_argument("--charges", type=Path, required=True, help="All-atom reference charges.npz keyed by SDF filename stem")
    parser.add_argument('--test-sdf', type=Path, help='Separate held-out SDF directory, never used for optimization or checkpoint selection')
    parser.add_argument('--test-charges', type=Path, help='Reference NPZ for --test-sdf')
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
    parser.add_argument('--wandb', nargs='?', const='online', default='disabled', choices=['disabled','offline','online'])
    parser.add_argument('--wandb-project', default='ccdc-molflae')
    parser.add_argument('--wandb-entity')
    parser.add_argument('--wandb-run-id', help='Resume tracking an existing W&B run (not optimizer state)')
    parser.add_argument('--reconstruction-molecules', type=int, default=32, help='Fixed training subset for latent-only decoding; 0 disables')
    parser.add_argument('--sample-steps', type=int, default=100)
    args = parser.parse_args(argv)
    if bool(args.test_sdf) != bool(args.test_charges):
        parser.error('--test-sdf and --test-charges must be supplied together')
    if args.reconstruction_molecules < 0 or args.sample_steps < 1:
        parser.error('Reconstruction count must be nonnegative and sampling steps positive')
    if min(args.epochs, args.batch_size, args.threads) < 1 or not math.isfinite(args.lr) or args.lr <= 0:
        parser.error("Epochs, batch size, threads, and learning rate must be positive")
    if not 0 <= args.val_fraction < 1 or not math.isfinite(args.max_grad_norm) or args.max_grad_norm <= 0:
        parser.error("Require 0 <= val-fraction < 1 and positive max-grad-norm")
    if not args.checkpoint.is_file():
        parser.error(f"Pretrained checkpoint not found: {args.checkpoint}; see README download instructions")
    os.environ["WANDB_MODE"] = args.wandb
    from _workflow.training import load_config, resolve_device, load_pretrained, molecular_data, validation_metrics
    import torch
    from torch_geometric.loader import DataLoader
    import wandb
    from _workflow.metrics import evaluate, latent_reconstruction

    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    records = structures(args.sdf)
    labels, verified = load_labels(args.charges, records)
    data = molecular_data(records, labels)
    test_records = structures(args.test_sdf) if args.test_sdf else []
    test_data = []
    if test_records:
        from rdkit import Chem
        def identities(items):
            return {Chem.MolToSmiles(Chem.RemoveHs(m), isomericSmiles=False) for _, _, m in items}
        if (identities(records) & identities(test_records)
                or {i.rstrip('0123456789') for i, _, _ in records}
                & {i.rstrip('0123456789') for i, _, _ in test_records}):
            raise ValueError('Training and test molecules overlap by identity or refcode family')
        test_labels, test_verified = load_labels(args.test_charges, test_records)
        test_data = molecular_data(test_records, test_labels)
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
    config["train"]["max_grad_norm"] = args.max_grad_norm
    config["train"]["batch_size"] = args.batch_size
    config["train"]["optimizer"]["lr"] = args.lr
    model, loaded = load_pretrained(config, args.checkpoint, device)
    model.external_logging = True
    if args.trainable == "head":
        for name, parameter in model.named_parameters():
            parameter.requires_grad_(name.startswith("decoder.charge_head."))
    model.configure_optimizers()
    train_loader = DataLoader([data[i] for i in train_indices], batch_size=args.batch_size, shuffle=True,
                             generator=torch.Generator().manual_seed(args.seed), num_workers=0)
    val_loader = DataLoader([data[i] for i in val_indices], batch_size=args.batch_size, num_workers=0)
    diagnostic_loader = DataLoader([data[i] for i in train_indices], batch_size=args.batch_size, num_workers=0)
    reconstruction_ids = train_indices[:args.reconstruction_molecules]
    reconstruction_loader = DataLoader([data[i] for i in reconstruction_ids], batch_size=min(args.batch_size,8), num_workers=0)
    test_loader = DataLoader(test_data, batch_size=args.batch_size, num_workers=0)
    test_reconstruction_loader = DataLoader(test_data[:args.reconstruction_molecules], batch_size=min(args.batch_size,8), num_workers=0)
    info = {"format": "ccdc-finetune-v1", "device": str(device), "seed": args.seed,
        "torch_version": str(torch.__version__), "batch_size": args.batch_size,
        "val_fraction": args.val_fraction, "max_grad_norm": args.max_grad_norm,
        "checkpoint": str(args.checkpoint.resolve()), "checkpoint_sha256": digest(args.checkpoint), **loaded,
        "charges": str(args.charges.resolve()), "charges_sha256": digest(args.charges),
        "labels_provenance_verified": verified, "charge_convention": "heavy_atom_absorb_h",
        "trainable": args.trainable, "epochs": args.epochs, "lr": args.lr,
        "wandb_mode":args.wandb, "sample_steps":args.sample_steps,
        "reconstruction_ids":[records[i][0] for i in reconstruction_ids],
        "gpu_name":torch.cuda.get_device_name(device) if device.type=='cuda' else None,
        "train_ids": [records[i][0] for i in train_indices], "validation_ids": [records[i][0] for i in val_indices],
        "source_sha256": {identifier: digest(path) for identifier, path, _ in records},
        "selection_metric": "validation_mae" if n_val else "training_loss"}
    if test_records:
        info.update(test_ids=[i for i, _, _ in test_records],
            test_source_sha256={i:digest(p) for i,p,_ in test_records},
            test_charges_sha256=digest(args.test_charges), test_labels_provenance_verified=test_verified)
    write_json(output / "run.json", info)
    write_json(output / "config.json", config)
    if not verified:
        print("Legacy labels: atom counts checked; no manifest available to verify source atom ordering.")
    best = float("inf")
    initial_head = {k: v.detach().clone() for k, v in model.decoder.charge_head.state_dict().items()}
    step = 0
    def synchronize():
        if device.type=='cuda': torch.cuda.synchronize(device)
    def diagnostics():
        result={'training_set':evaluate(model,diagnostic_loader,device,args.seed+1000)}
        if n_val: result['validation_set']=evaluate(model,val_loader,device,args.seed+1001)
        if reconstruction_ids:
            result['latent_only']=latent_reconstruction(model,reconstruction_loader,device,args.sample_steps,args.seed+2000)
        if test_records:
            result['test_set']=evaluate(model,test_loader,device,args.seed+1002)
            if args.reconstruction_molecules:
                result['test_latent_only']=latent_reconstruction(model,test_reconstruction_loader,device,args.sample_steps,args.seed+2001)
        return result
    tracking_config={k:info[k] for k in ('device','seed','batch_size','epochs','lr','trainable','sample_steps','checkpoint_sha256','charges_sha256')}
    tracking_config.update(train_molecules=len(train_indices), validation_molecules=len(val_indices), test_molecules=len(test_records))
    run_start=time.monotonic()
    with wandb.init(mode=args.wandb,project=args.wandb_project,entity=args.wandb_entity,
            id=args.wandb_run_id,resume='allow' if args.wandb_run_id else None,
            dir=str(output),save_code=False,
            settings=wandb.Settings(disable_git=True)) as run:
        run.config.update(tracking_config, allow_val_change=True)
        if args.wandb!='disabled':
            write_json(output/'wandb.json',{'id':run.id,'url':run.url,'mode':args.wandb})
            run.summary['phase']='baseline'
            run.define_metric('evaluation_epoch')
            run.define_metric('evaluation/*', step_metric='evaluation_epoch')
        baseline=diagnostics()
        write_json(output/'before.json',baseline)
        run.log({'evaluation_epoch':0, **wandb_diagnostic_metrics(baseline)})
        if args.wandb!='disabled': run.summary['phase']='training'
        with (output/'metrics.jsonl').open('w') as metrics_file, (output/'steps.jsonl').open('w') as steps_file:
            for epoch in range(1,args.epochs+1):
                model.train(); totals=defaultdict(float); graphs=0
                if device.type=='cuda': torch.cuda.reset_peak_memory_stats(device)
                synchronize(); epoch_start=time.monotonic()
                for batch in train_loader:
                    batch=batch.to(device)
                    model.optim.zero_grad(set_to_none=True)
                    loss=model.training_step(batch,step)
                    if loss is None or not torch.isfinite(loss): raise ValueError(f'Non-finite loss at step {step}')
                    loss.backward()
                    grad_norm=torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],
                        args.max_grad_norm,error_if_nonfinite=True)
                    model.optim.step()
                    for key,value in model.last_loss_metrics.items(): totals[key]+=value*batch.num_graphs
                    graphs+=batch.num_graphs; step+=1
                    logged={'epoch':epoch,'optimizer_step':step,'gradient_norm':float(grad_norm),
                        'lr':model.optim.param_groups[0]['lr'],**model.last_loss_metrics}
                    steps_file.write(json.dumps(logged,allow_nan=False)+'\n'); steps_file.flush()
                    run.log(wandb_training_metrics(logged, 'train'))
                synchronize(); epoch_seconds=time.monotonic()-epoch_start
                performance={'epoch_seconds':epoch_seconds,'molecules_per_second':graphs/epoch_seconds,
                    'peak_allocated_GiB':torch.cuda.max_memory_allocated(device)/2**30 if device.type=='cuda' else 0,
                    'peak_reserved_GiB':torch.cuda.max_memory_reserved(device)/2**30 if device.type=='cuda' else 0}
                measured=diagnostics()
                metrics={'epoch':epoch,'step':step,'training_loss':totals['loss']/graphs,
                    'training_components':{k:v/graphs for k,v in totals.items()},'performance':performance,
                    'diagnostics':measured}
                metrics_file.write(json.dumps(metrics,allow_nan=False)+'\n'); metrics_file.flush()
                print(json.dumps({'epoch':epoch,'training_loss':metrics['training_loss'],**performance}),flush=True)
                write_json(output/'after.json',measured)
                run.log(wandb_training_metrics(metrics['training_components'], 'epoch'))
                run.log({'evaluation_epoch':epoch, **wandb_diagnostic_metrics(measured)})
                score=measured['validation_set']['given_geometry_charges']['mae_e'] if n_val else metrics['training_loss']
                payload={'format':'ccdc-finetune-v1','state_dict':model.state_dict(),'config':config,
                    'epoch':epoch,'global_step':step,'optimizer_state':model.optim.state_dict(),
                    'charge_convention':'heavy_atom_absorb_h','metrics':metrics,'source':info}
                torch.save(payload,output/'last.ckpt')
                if score<best:
                    best=score; torch.save(payload,output/'best.ckpt')
                model.train_losses.clear()
        if args.wandb!='disabled':
            run.summary.update(wandb_diagnostic_metrics(measured, 'final'))
            run.summary.update({f'performance/{key}': value for key, value in performance.items()})
            run.summary['phase']='complete'
            run.summary['total_seconds']=time.monotonic()-run_start
    changed = any(not torch.equal(v, model.decoder.charge_head.state_dict()[k]) for k, v in initial_head.items())
    if not changed:
        raise ValueError("No charge-head parameters changed during training")
    write_json(output / "result.json", {"steps": step, "charge_head_updated": changed,
        "elapsed_seconds":time.monotonic()-run_start, "performance":performance, "best_score": best, "selection_metric": info["selection_metric"], "output": str(output)})
    print(f"Fine-tuning complete. Checkpoints and metrics: {output}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (ValueError, OSError, ImportError, RuntimeError) as exc:
        sys.exit(str(exc))
