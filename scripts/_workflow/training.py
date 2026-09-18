"""Model/data adapters for notebook-free MolFLAE fine-tuning."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "MolFLAE"))

import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.data import Data

from model.train_loop import TrainLoopCharges, center_pos, MAP_ATOM_TYPE_ONLY_TO_INDEX
from utils.config import load_config
from utils.device import resolve_device


def molecular_data(records, labels):
    """Preserve heavy-atom order and absorb bonded H charges as in the notebook."""
    data = []
    for identifier, _, mol in records:
        q = labels[identifier].copy()
        for atom in mol.GetAtoms():
            if atom.GetAtomicNum() == 1:
                neighbors = atom.GetNeighbors()
                if len(neighbors) != 1 or neighbors[0].GetAtomicNum() == 1:
                    raise ValueError(f"Cannot absorb hydrogen charge in {identifier}")
                q[neighbors[0].GetIdx()] += q[atom.GetIdx()]
                q[atom.GetIdx()] = 0
        indices = [a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() != 1]
        numbers = [mol.GetAtomWithIdx(i).GetAtomicNum() for i in indices]
        if not numbers or set(numbers) - set(MAP_ATOM_TYPE_ONLY_TO_INDEX):
            raise ValueError(f"Unsupported heavy atoms in {identifier}: {numbers}")
        if not np.isclose(q[indices].sum(), labels[identifier].sum()):
            raise ValueError(f"Hydrogen removal changed total charge in {identifier}")
        data.append(Data(h=torch.tensor(numbers, dtype=torch.long).view(-1, 1),
            x=torch.tensor(mol.GetConformer().GetPositions()[indices], dtype=torch.float32),
            charges=torch.tensor(q[indices], dtype=torch.float32).view(-1, 1),
            molecule_id=identifier))
    return data


def load_pretrained(config, checkpoint, device):
    model = TrainLoopCharges(config, device=device)
    payload = torch.load(checkpoint, map_location=device, weights_only=True)
    state = payload["state_dict"]
    expected = {k for k in model.state_dict() if k.startswith("decoder.charge_head.")}
    result = model.load_state_dict(state, strict=False)
    missing = set(result.missing_keys)
    if result.unexpected_keys or missing not in (set(), expected):
        raise ValueError(f"Incompatible checkpoint: missing={sorted(missing)}, unexpected={result.unexpected_keys}")
    return model, {"new_charge_head": bool(missing), "loaded_tensors": len(state)}


def charge_predictions(model, batch):
    """Deterministic charge evaluation on the supplied geometry (no molecule generation)."""
    x, _ = center_pos(batch.x, batch.batch, mode=model.cfg["decoder_config"]["center_pos_mode"])
    indices = torch.tensor([MAP_ATOM_TYPE_ONLY_TO_INDEX[int(n)] for n in batch.h.flatten().tolist()], device=x.device)
    types = F.one_hot(indices, model.cfg["encoder_config"]["ligand_v_dim"]).float()
    zh, zx, graphs, _, _ = model.encode(types, x, batch.batch, deterministic=True)
    time = torch.ones((len(x), 1), device=x.device, dtype=x.dtype)
    gamma = 1 - model.decoder.sigma1_coord.square()
    _, _, q = model.decoder.interdependency_modeling(time=time,
        protein_pos=zx, protein_v=zh, batch_protein=graphs,
        theta_h_t=types, mu_pos_t=gamma * x, batch_ligand=batch.batch,
        gamma_coord=gamma.expand_as(time))
    return q


def validation_metrics(model, loader, device):
    model.eval()
    absolute, squared, count = 0.0, 0.0, 0
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            errors = charge_predictions(model, batch) - batch.charges
            if not torch.isfinite(errors).all():
                raise ValueError("Non-finite validation predictions")
            absolute += errors.abs().sum().item()
            squared += errors.square().sum().item()
            count += errors.numel()
    return {"mae": absolute / count, "rmse": (squared / count) ** 0.5} if count else None
