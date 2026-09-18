"""Separate stochastic training objectives, supplied-geometry charges, and latent-only decoding."""
from collections import defaultdict
import time
import numpy as np
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from model.train_loop import center_pos, MAP_ATOM_TYPE_ONLY_TO_INDEX


def inputs(model, batch):
    x, _ = center_pos(batch.x, batch.batch, mode=model.cfg['decoder_config']['center_pos_mode'])
    indices = torch.tensor([MAP_ATOM_TYPE_ONLY_TO_INDEX[int(n)] for n in batch.h.flatten().tolist()],device=x.device)
    types = F.one_hot(indices,model.decoder.num_classes).float()
    return x,indices,types


def charge_statistics(predicted, target, graph, element):
    error=predicted.flatten()-target.flatten()
    if not torch.isfinite(error).all(): raise ValueError('Non-finite charge predictions')
    groups=int(graph.max())+1
    sums=torch.zeros(groups,device=error.device).index_add_(0,graph,error)
    counts=torch.bincount(graph,minlength=groups)
    mol_mse=torch.zeros(groups,device=error.device).index_add_(0,graph,error.square())/counts
    return {'mae_e':error.abs().mean().item(),'rmse_e':error.square().mean().sqrt().item(),
        'mse_e2':error.square().mean().item(), 'molecule_mean_mse_e2':mol_mse.mean().item(),
        'total_charge_mae_e':sums.abs().mean().item(),'total_charge_mse_e2':sums.square().mean().item(),
        'p95_absolute_error_e':torch.quantile(error.abs(),0.95).item(),
        'per_element_mae_e':{str(int(z)):error[element.flatten()==z].abs().mean().item() for z in element.unique()}}


def evaluate(model, loader, device, seed=123):
    """Fixed-seed objectives and deterministic charge predictions with known geometry/types."""
    model.eval()
    totals=defaultdict(float); graphs=0; predictions=[]; labels=[]; elements=[]; memberships=[]
    device = torch.device(device)
    cuda_devices=[device.index if device.index is not None else torch.cuda.current_device()] if device.type=='cuda' else []
    with torch.random.fork_rng(devices=cuda_devices), torch.no_grad():
        torch.manual_seed(seed)
        for batch in loader:
            batch=batch.to(device)
            loss=model.training_step(batch,0)
            if loss is None or not torch.isfinite(loss): raise ValueError('Non-finite evaluation loss')
            for key,value in model.last_loss_metrics.items(): totals[key]+=value*batch.num_graphs
            x,indices,types=inputs(model,batch)
            zh,zx,global_batch,_,_=model.encode(types,x,batch.batch,deterministic=True)
            t=torch.ones((len(x),1),device=x.device)
            gamma=1-model.decoder.sigma1_coord.square()
            _,_,q=model.decoder.interdependency_modeling(time=t,protein_pos=zx,protein_v=zh,
                batch_protein=global_batch,theta_h_t=types,mu_pos_t=gamma*x,batch_ligand=batch.batch,
                gamma_coord=gamma.expand_as(t))
            predictions.append(q.cpu()); labels.append(batch.charges.cpu()); elements.append(batch.h.cpu())
            memberships.append(batch.batch.cpu()+graphs); graphs+=batch.num_graphs
    model.train_losses.clear()
    if not graphs: return None
    q,y,z,b=map(torch.cat,(predictions,labels,elements,memberships))
    stats=charge_statistics(q,y,b,z)
    stats['weighted_charge_loss']=model.decoder.charge_loss_weight*stats['molecule_mean_mse_e2']+model.decoder.total_charge_weight*stats['total_charge_mse_e2']
    return {'molecules':graphs,'atoms':len(y),'objectives':{k:v/graphs for k,v in totals.items()},
        'given_geometry_charges':stats,'zero_charge_baseline':charge_statistics(torch.zeros_like(y),y,b,z)}


def latent_reconstruction(model, loader, device, sample_steps=100, seed=456):
    """Decode from latents and atom counts only; match unordered atoms by spatial distance.

    Coordinates are in the encoder's centered reference frame. No ground-truth
    atom types or coordinates are passed to the decoder, and no rotation is fitted.
    """
    model.eval(); rows=[]
    device = torch.device(device)
    cuda_devices=[device.index if device.index is not None else torch.cuda.current_device()] if device.type=='cuda' else []
    start=time.monotonic()
    with torch.random.fork_rng(devices=cuda_devices), torch.no_grad():
        torch.manual_seed(seed)
        for batch in loader:
            batch=batch.to(device)
            x,indices,types=inputs(model,batch)
            zh,zx,global_batch,_,_=model.encode(types,x,batch.batch,deterministic=True)
            theta,trajectory,y=model.decoder.sample(protein_pos=zx,protein_v=zh,batch_protein=global_batch,
                batch_ligand=batch.batch,n_nodes=len(x),sample_steps=sample_steps,desc='latent-only reconstruction')
            positions,probabilities,q,_=trajectory[-1]
            if not all(torch.isfinite(v).all() for v in (positions,probabilities,q)):
                raise ValueError('Non-finite latent-only reconstruction')
            for i in range(batch.num_graphs):
                mask=batch.batch==i
                target=x[mask].cpu().numpy(); predicted=positions[mask].cpu().numpy()
                predicted=predicted-predicted.mean(axis=0)
                cost=((target[:,None,:]-predicted[None,:,:])**2).sum(axis=-1)
                a,b=linear_sum_assignment(cost)
                predicted_types=probabilities[mask].argmax(-1).cpu().numpy()[b]
                target_types=indices[mask].cpu().numpy()[a]
                errors=q[mask].flatten().cpu().numpy()[b]-batch.charges[mask].flatten().cpu().numpy()[a]
                rows.append({'identifier':batch.molecule_id[i],'atoms':len(a),
                    'position_squared_error_A2':float(cost[a,b].sum()),
                    'correct_atom_types':int((predicted_types==target_types).sum()),
                    'charge_absolute_error_e':float(np.abs(errors).sum()),
                    'charge_squared_error_e2':float((errors**2).sum())})
            del theta,trajectory,y
    n=sum(r['atoms'] for r in rows)
    return {'molecules':len(rows),'atoms':n,'sample_steps':sample_steps,
        'matched_position_rmse_A':(sum(r['position_squared_error_A2'] for r in rows)/n)**0.5,
        'matched_atom_type_accuracy':sum(r['correct_atom_types'] for r in rows)/n,
        'matched_charge_mae_e':sum(r['charge_absolute_error_e'] for r in rows)/n,
        'matched_charge_rmse_e':(sum(r['charge_squared_error_e2'] for r in rows)/n)**0.5,
        'elapsed_seconds':time.monotonic()-start,'per_molecule':rows,
        'matching':'position-only Hungarian assignment in centered encoder frame; atom count supplied'}
