"""Run the checked-in encoder and one CPU charge-training step, without W&B uploads."""

import json
import os
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT / "MolFLAE")
sys.path.insert(0, str(ROOT / "MolFLAE"))
os.environ["WANDB_MODE"] = "disabled"

import numpy as np
import torch
from torch import nn
from torch_geometric.data import Batch, Data
from rdkit import Chem
import wandb

from model.encoder_standalone_cpu import Encoder, molecule_to_latent
from model.train_loop import TrainLoopCharges
from utils.config import load_config
from utils.data_loading import MAP_ATOM_TYPE_ONLY_TO_INDEX


def main():
    torch.set_num_threads(2)
    torch.manual_seed(0)
    output_root = ROOT / "runs"
    output_root.mkdir(exist_ok=True)
    output = Path(tempfile.mkdtemp(prefix="ml-", dir=output_root))
    cfg = load_config("config.yaml")
    cfg["evaluation"]["save_dir"] = str(output)

    mol = next(iter(Chem.SDMolSupplier(str(ROOT / "csd_mol.sdf"), removeHs=False)))
    assert mol is not None and mol.GetNumConformers() > 0
    with np.load(ROOT / "resp_charges.npz", allow_pickle=False) as archive:
        charges = archive["csd_mol"].copy()
    assert charges.shape == (mol.GetNumAtoms(),)
    original_total = charges.sum()
    # Use the notebook's implicit-H convention, preserving total molecular charge.
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() == 1:
            neighbors = atom.GetNeighbors()
            assert len(neighbors) == 1 and neighbors[0].GetAtomicNum() != 1
            charges[neighbors[0].GetIdx()] += charges[atom.GetIdx()]
            charges[atom.GetIdx()] = 0
    numbers = np.array([a.GetAtomicNum() for a in mol.GetAtoms()])
    heavy = numbers != 1
    numbers = numbers[heavy]
    x = torch.tensor(mol.GetConformer().GetPositions()[heavy], dtype=torch.float32)
    q = torch.tensor(charges[heavy], dtype=torch.float32).view(-1, 1)
    assert np.isclose(charges[heavy].sum(), original_total)
    indices = torch.tensor([MAP_ATOM_TYPE_ONLY_TO_INDEX[int(n)] for n in numbers])

    encoder = Encoder(**cfg["encoder_config"])
    encoder.load_state_dict(torch.load("weights/encoder_weights.pth", map_location="cpu", weights_only=True))
    for name, filename, out_dim in (
        ("Wh_mu", "encoder_weights_KL.pth", cfg["optimal_layer_config"]["latent_dim"]),
        ("Wh_log_var", "encoder_weights_KL_Wh_log_var.pth", cfg["optimal_layer_config"]["latent_dim"]),
        ("Wx_log_var", "encoder_weights_KL_Wx_log_var.pth", 1),
    ):
        layer = nn.Linear(cfg["encoder_config"]["hidden_dim"], out_dim)
        layer.load_state_dict(torch.load(f"weights/{filename}", map_location="cpu", weights_only=True))
        setattr(encoder, name, layer)
    encoder.eval()
    with torch.no_grad():
        zh, zx, graph = molecule_to_latent(encoder, {"h": indices, "x": x})
    assert torch.isfinite(zh).all() and torch.isfinite(zx).all()
    np.savez_compressed(output / "latent.npz", Zh=zh.numpy(), Zx=zx.numpy(), batch=graph.numpy())

    # Exercise the actual training wrapper with correctly aligned example labels.
    batch = Batch.from_data_list([Data(
        h=torch.tensor(numbers, dtype=torch.long).view(-1, 1), x=x, charges=q,
    )])
    with wandb.init(mode="disabled"):
        model = TrainLoopCharges(cfg)
        model.configure_optimizers()
        model.train()
        model.optim.zero_grad()
        loss = model.training_step(batch, 0)
        assert torch.isfinite(loss)
        loss.backward()
        grads = [p.grad for p in model.decoder.charge_head.parameters()]
        assert all(g is not None and torch.isfinite(g).all() for g in grads)
        assert any(torch.count_nonzero(g) for g in grads)
        before = [p.detach().clone() for p in model.decoder.charge_head.parameters()]
        model.optim.step()
        assert any(not torch.equal(a, b) for a, b in zip(before, model.decoder.charge_head.parameters()))
    result = {
        "torch": torch.__version__, "device": "cpu", "heavy_atoms": len(numbers),
        "total_charge": float(original_total), "Zh_shape": list(zh.shape),
        "Zx_shape": list(zx.shape), "training_loss": float(loss.detach()),
        "charge_head_updated": True, "output": str(output),
    }
    (output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
