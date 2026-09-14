import copy
import numpy as np
import torch
from types import SimpleNamespace
from time import time
import torch.nn.functional as F
import pytorch_lightning as pl
from torch_geometric.data import Batch
from torch_scatter import scatter_mean, scatter_sum
import os
from model.bfn4sbdd import BFN_charge, BFN4SBDDScoreModel
from utils.device import resolve_device
from model.encoder import Encoder
import utils.atom_num as atom_num
import datetime
import json
from utils.train import get_optimizer, get_scheduler
import wandb
from utils.build_mol import MoleculeBuilder
from rdkit import Chem
import torch.nn as nn

MAP_ATOM_TYPE_ONLY_TO_INDEX = {
    6: 0,
    7: 1,
    8: 2,
    9: 3,
    15: 4,
    16: 5,
    17: 6,
    35: 7,
    53: 8,
}
MAP_INDEX_TO_ATOM_TYPE_ONLY = {v: k for k, v in MAP_ATOM_TYPE_ONLY_TO_INDEX.items()}


def center_pos(ligand_pos, batch_ligand, mode: bool = True):
    if not mode:
        offset = 0.0
    else:
        offset = scatter_mean(ligand_pos, batch_ligand, dim=0)
        ligand_pos = ligand_pos - offset[batch_ligand]

    return ligand_pos, offset


def dict_to_namespace(d):
    if not isinstance(d, dict):
        return d
    return SimpleNamespace(**{k: dict_to_namespace(v) for k, v in d.items()})


class TrainLoop(pl.LightningModule):
    def __init__(self, cfg, device=None):
        """Training loop wrapper

        Args:
            cfg (dict): Configuration dictionary (encoder/decoder/training/eval).
        """
        super().__init__()
        self.cfg = cfg

        # Instantiate encoder and decoder modules
        self.encoder = Encoder(**self.cfg['encoder_config'])
        decoder_config = dict(self.cfg['decoder_config'])
        legacy_device = decoder_config.pop('device', None)
        selected_device = resolve_device(device if device is not None else legacy_device, self.cfg)
        self.decoder = BFN4SBDDScoreModel(**decoder_config)

        # Linear layers used by the KL / latent parametrization
        self.Wh_mu = nn.Linear(
            self.cfg['encoder_config']['hidden_dim'],
            self.cfg['optimal_layer_config']['latent_dim']
        )
        self.Wh_log_var = nn.Linear(
            self.cfg['encoder_config']['hidden_dim'],
            self.cfg['optimal_layer_config']['latent_dim']
        )
        # Must be isotropic (scalar per node) to preserve equivariance
        self.Wx_log_var = nn.Linear(self.cfg['encoder_config']['hidden_dim'], 1)

        # bookkeeping / diagnostics
        self.time_records = np.zeros(6)
        self.encoder_config = self.cfg['encoder_config']
        self.decoder_config = self.cfg['decoder_config']

        self.train_losses = []
        self.log_time = False

        # print parameter counts for encoder / KL layer / decoder
        self.print_model_params()

        # directory for evaluation outputs
        self.save_dir = cfg['evaluation']['save_dir']
        os.makedirs(self.save_dir, exist_ok=True)

        self.test_result = []
        self.to(selected_device)

    def print_model_params(self):
        """Print number of trainable parameters"""
        def count_params(module):
            return sum(p.numel() for p in module.parameters() if p.requires_grad)

        encoder_params = count_params(self.encoder)
        optimal_layer_params = (
            count_params(self.Wh_log_var)
            + count_params(self.Wh_mu)
            + count_params(self.Wx_log_var)
        )
        decoder_params = count_params(self.decoder)
        total_params = encoder_params + decoder_params + optimal_layer_params

        print("\n" + "=" * 50)
        print(f"Encoder params: {encoder_params/1e6:.2f}M")
        print(f"KL layer params: {optimal_layer_params/1e6:.2f}M")
        print(f"Decoder params: {decoder_params/1e6:.2f}M")
        print(f"total params: {total_params/1e6:.2f}M")
        print("=" * 50 + "\n")

    def forward(self, x):
        """Forward is unused here. Kept as placeholder."""
        pass

    def encode(self, one_hot_h, x, batch_ligand, deterministic=False):
        """Encode input to latent representations and compute KL losses.

        Args:
            one_hot_h (Tensor): one-hot encoded ligand node types, shape [N, K].
            x (Tensor): node positions, shape [N, 3].
            batch_ligand (Tensor): batch index per node, shape [N].
            deterministic (bool): if True, do not sample from latent distributions.

        Returns:
            Zh_sampled (Tensor): sampled / deterministic latent for node attributes.
            Zx_sampled (Tensor): sampled / deterministic latent for node positions.
            global_batch (Tensor): batch indices corresponding to global nodes.
            Zh_kl_loss (Tensor): KL loss for Zh (attribute latent).
            Zx_kl_loss (Tensor): KL loss for Zx (position latent).
        """
        # encoder returns global node features, global positions, and batch mapping
        global_h, global_x, global_batch = self.encoder(one_hot_h, x, batch_ligand)

        # Parameterize latent Gaussian for Zh (attribute latent)
        Zh_mu = self.Wh_mu(global_h)
        Zh_log_var = -torch.abs(self.Wh_log_var(global_h))  # ensure non-positive values

        # For Zx, use global_x as mean and predict a scalar log-variance per global node
        Zx_mu = global_x.clone()

        # clamp log variance to avoid excessively large variance
        upper = torch.log(
            torch.tensor(self.cfg['train']['kl_loss']['sigma2']**2,
                         device=Zx_mu.device, dtype=Zx_mu.dtype)
        )
        raw = self.Wx_log_var(global_h).expand_as(Zx_mu)
        Zx_log_var = torch.clamp(raw, max=upper)

        # number of graphs in the batch (unique batch indices)
        data_size = torch.unique(global_batch).size(0)

        # KL divergences (averaged per-element and per-dimension)
        Zh_kl_loss = -0.5 * torch.sum(
            1.0 + Zh_log_var - Zh_mu * Zh_mu - torch.exp(Zh_log_var)
        ) / (data_size * Zh_mu.shape[-1])

        Zx_kl_loss = -0.5 * torch.sum(
            1.0 + Zx_log_var - (Zx_mu * Zx_mu + torch.exp(Zx_log_var)) / (self.cfg['train']['kl_loss']['sigma2'])**2
        ) / (data_size * Zx_mu.shape[-1])

        # Reparameterization: sample if not deterministic
        Zh_sampled = Zh_mu if deterministic else Zh_mu + torch.exp(Zh_log_var / 2) * torch.randn_like(Zh_mu)
        Zx_sampled = Zx_mu if deterministic else Zx_mu + torch.exp(Zx_log_var / 2) * torch.randn_like(Zx_mu)

        return Zh_sampled, Zx_sampled, global_batch, Zh_kl_loss, Zx_kl_loss

    def training_step(self, batch, batch_idx):
        """Single training step: encode, compute decoder losses, log and return loss."""
        t1 = time()

        h = batch['h']           # original atom types [N_node, 1]
        x = batch['x']           # positions [N_node, 3]
        batch_ligand = batch['batch']  # batch indices per node [N_node]

        t2 = time()
        # number of graphs in this batch
        num_graphs = batch_ligand.max().item() + 1

        # center ligand positions according to configured mode
        x, _ = center_pos(
            ligand_pos=x,
            batch_ligand=batch_ligand,
            mode=self.cfg['decoder_config']['center_pos_mode']
        )

        t3 = time()

        # sample a time t for each graph, then select per-node by batch index
        t = torch.rand([num_graphs, 1], dtype=x.dtype, device=x.device).index_select(0, batch_ligand)

        # optional clamping if decoder uses continuous t
        if not self.cfg['decoder_config']['use_discrete_t'] and not self.cfg['decoder_config']['destination_prediction']:
            t = torch.clamp(t, min=self.decoder.t_min)

        t4 = time()

        # map atom types to indices and one-hot encode
        h = [MAP_ATOM_TYPE_ONLY_TO_INDEX[i.item()] for i in h]
        h = torch.tensor(h, dtype=torch.long).to(x.device)
        K = self.cfg['encoder_config']['ligand_v_dim']
        one_hot_h = F.one_hot(h, K).float()  # [N, K]

        # encode to obtain latent samples and KL losses
        Zh_sampled, Zx_sampled, global_batch, Zh_kl_loss, Zx_kl_loss = self.encode(
            one_hot_h, x, batch_ligand, deterministic=False
        )

        # decoder loss for this step (returns c_loss, d_loss, discretised_loss)
        c_loss, d_loss, discretised_loss = self.decoder.loss_one_step(
            t,
            protein_pos=Zx_sampled,
            protein_v=Zh_sampled,
            batch_protein=global_batch,
            ligand_pos=x,
            ligand_v=h,
            batch_ligand=batch_ligand,
        )

        # reconstruction and KL combined loss (mean over batch)
        recon_loss = torch.mean(
            self.cfg['train']['recon_loss']['c_loss_weight'] * c_loss
            + self.cfg['train']['recon_loss']['d_loss_weight'] * d_loss
            + discretised_loss
        )
        kl_loss = torch.mean(
            self.cfg['train']['kl_loss']['Zh_kl_loss_weight'] * Zh_kl_loss
            + self.cfg['train']['kl_loss']['Zx_kl_loss_weight'] * Zx_kl_loss
        )
        loss = (
            self.cfg['train']['recon_loss']['recon_loss_weight'] * recon_loss
            + self.cfg['train']['kl_loss']['kl_loss_weight'] * kl_loss
        )

        # logging to W&B and Lightning
        wandb.log({
            'lr': self.get_last_lr(),
            'train_loss': loss.item(),
            'train_recon_loss': recon_loss.item(),
            'train_kl_loss': kl_loss.item()
        })

        t5 = time()

        self.log_dict(
            {
                'lr': self.get_last_lr(),
                'train_loss': loss.item(),
                'recon_loss': recon_loss.item(),
                'kl_loss': kl_loss.item()
            },
            on_step=True,
            prog_bar=True,
            batch_size=self.cfg['train']['batch_size'],
        )

        # skip updates when loss is not finite
        if not torch.isfinite(loss):
            return None

        self.train_losses.append(loss.clone().detach().cpu())

        t0 = time()

        # optional timing diagnostics
        if self.log_time:
            self.time_records = np.vstack((self.time_records, [t0, t1, t2, t3, t4, t5]))
            print(f'step total time: {self.time_records[-1, 0] - self.time_records[-1, 1]}, batch size: {num_graphs}')
            print(f'\tpl call & data access: {self.time_records[-1, 1] - self.time_records[-2, 0]}')
            print(f'\tunwrap data: {self.time_records[-1, 2] - self.time_records[-1, 1]}')
            print(f'\tadd noise & center pos: {self.time_records[-1, 3] - self.time_records[-1, 2]}')
            print(f'\tsample t: {self.time_records[-1, 4] - self.time_records[-1, 3]}')
            print(f'\tget loss: {self.time_records[-1, 5] - self.time_records[-1, 4]}')
            print(f'\tlogging: {self.time_records[-1, 0] - self.time_records[-1, 5]}')

        return loss

    def validation_step(self, batch, batch_idx):
        """Validation: deterministic encode, compute losses, sample and compute similarity metrics."""
        original_h = batch['h']
        original_x = batch['x']
        batch_ligand = batch['batch']

        # center ligand positions for evaluation
        x, _ = center_pos(
            ligand_pos=original_x,
            batch_ligand=batch_ligand,
            mode=self.cfg['decoder_config']['center_pos_mode']
        )

        num_graphs = batch_ligand.max().item() + 1

        # sample time t per graph, clamp if needed
        t = torch.rand([num_graphs, 1], dtype=x.dtype, device=x.device).index_select(0, batch_ligand)
        if not self.cfg['decoder_config']['use_discrete_t'] and not self.cfg['decoder_config']['destination_prediction']:
            t = torch.clamp(t, min=self.decoder.t_min)

        # map atom types and one-hot encode
        h = [MAP_ATOM_TYPE_ONLY_TO_INDEX[i.item()] for i in original_h]
        h = torch.tensor(h, dtype=torch.long).to(x.device)
        K = self.cfg['encoder_config']['ligand_v_dim']
        one_hot_h = F.one_hot(h, K).float()  # [N, K]

        # deterministic encode for validation
        Zh_sampled, Zx_sampled, global_batch, Zh_kl_loss, Zx_kl_loss = self.encode(
            one_hot_h, x, batch_ligand, deterministic=True
        )

        # compute decoder losses
        c_loss, d_loss, discretised_loss = self.decoder.loss_one_step(
            t,
            protein_pos=Zx_sampled,
            protein_v=Zh_sampled,
            batch_protein=global_batch,
            ligand_pos=x,
            ligand_v=h,
            batch_ligand=batch_ligand,
        )

        recon_loss = torch.mean(
            self.cfg['train']['recon_loss']['c_loss_weight'] * c_loss
            + self.cfg['train']['recon_loss']['d_loss_weight'] * d_loss
            + discretised_loss
        )
        kl_loss = torch.mean(
            self.cfg['train']['kl_loss']['Zh_kl_loss_weight'] * Zh_kl_loss
            + self.cfg['train']['kl_loss']['Zx_kl_loss_weight'] * Zx_kl_loss
        )
        loss = recon_loss + kl_loss

        # generate deterministic samples for similarity evaluation
        generated_data = self.shared_sampling_step(batch, batch_idx, sample_num_atoms='ref', desc='Val', deterministic=True)

        molecule_builder = MoleculeBuilder()

        generated_h = torch.tensor(generated_data['h'], dtype=torch.long).to(x.device)
        generated_x = generated_data['x'].to(x.device)
        generated_batch = generated_data['batch'].to(x.device)

        unique_batches = torch.unique(batch_ligand)

        similarities = []
        for batch_idx_val in unique_batches:
            original_mask = batch_ligand == batch_idx_val
            original_atoms = original_h[original_mask].cpu().numpy().tolist()
            original_coords = original_x[original_mask].cpu().numpy()

            generated_mask = generated_batch == batch_idx_val
            generated_atoms = generated_h[generated_mask].cpu().numpy().tolist()
            generated_coords = generated_x[generated_mask].cpu().numpy()

            original_mol = molecule_builder.build_mol(original_coords, original_atoms)
            generated_mol = molecule_builder.build_mol(generated_coords, generated_atoms)

            if original_mol is None or generated_mol is None:
                print(f"Warning: Invalid molecule for batch_idx_val {batch_idx_val.item()}")
                continue

            similarity = molecule_builder.compute_iou(original_mol, generated_mol)
            similarities.append(similarity)

        avg_similarity = torch.tensor(similarities).mean().item() if similarities else 0.0

        # log metrics
        wandb.log({
            'val_loss': loss.item(),
            'val_recon_loss': recon_loss.item(),
            'val_kl_loss': kl_loss.item(),
            'val_similarity': avg_similarity,
        })
        self.log_dict({
            'val_loss': loss.item(),
            'val_similarity': avg_similarity,
        },
            prog_bar=True,
            logger=True,
            on_step=True,
            sync_dist=True,
            batch_size=self.cfg['evaluation']['batch_size'],
        )

        return loss

    def shared_sampling_step(self, batch, batch_idx, sample_num_atoms, desc='', deterministic=True):
        """Shared sampling routine used in validation and testing.

        Args:
            batch (dict): input batch with 'h', 'x', 'batch'
            sample_num_atoms (str): 'prior' or 'ref' (or other modes raise ValueError)
            desc (str): description passed to decoder.sample for logging
            deterministic (bool): whether to use deterministic latent samples

        Returns:
            out_data (dict): {'h': predicted atom types (list), 'x': positions Tensor, 'batch': batch Tensor}
        """
        h = batch['h']
        x = batch['x']
        batch_ligand = batch['batch']

        num_graphs = batch_ligand.max().item() + 1  # number of molecules in this batch
        n_nodes = batch_ligand.size(0)  # total nodes

        # center positions and get offset for restoring coordinates later
        x, offset = center_pos(ligand_pos=x, batch_ligand=batch_ligand, mode=True)

        # decide per-graph atom counts
        if sample_num_atoms == 'prior':
            ligand_num_atoms = []
            for data_id in range(len(batch)):
                data = batch[data_id]
                pocket_size = atom_num.get_space_size(data.protein_pos.detach().cpu().numpy() * self.cfg['data']['normalizer_dict']['pos'])
                ligand_num_atoms.append(atom_num.sample_atom_num(pocket_size).astype(int))
            batch_ligand = torch.repeat_interleave(torch.arange(len(batch)), torch.tensor(ligand_num_atoms)).to(x.device)
            ligand_num_atoms = torch.tensor(ligand_num_atoms, dtype=torch.long, device=x.device)
        elif sample_num_atoms == 'ref':
            batch_ligand = batch_ligand
            ligand_num_atoms = scatter_sum(torch.ones_like(batch_ligand), batch_ligand, dim=0).to(x.device)
        else:
            raise ValueError(f"sample_num_atoms mode: {sample_num_atoms} not supported")

        # cumulative counts (useful for indexing if needed)
        ligand_cum_atoms = torch.cat([
            torch.tensor([0], dtype=torch.long, device=x.device),
            ligand_num_atoms.cumsum(dim=0)
        ])

        # map atom types to indices and one-hot encode
        h = [MAP_ATOM_TYPE_ONLY_TO_INDEX[i.item()] for i in h]
        h = torch.tensor(h, dtype=torch.long).to(x.device)
        K = self.cfg['encoder_config']['ligand_v_dim']
        h = F.one_hot(h, K).float()  # [N, K]

        # encode (deterministic or sampled)
        global_nodes, global_position, global_batch, Zh_kl_loss, Zx_kl_loss = self.encode(h, x, batch_ligand, deterministic=deterministic)

        # sample molecule chain from decoder
        theta_chain, sample_chain, y_chain = self.decoder.sample(
            protein_pos=global_position,
            protein_v=global_nodes,
            batch_protein=global_batch,
            batch_ligand=batch_ligand,
            sample_steps=self.cfg['evaluation']['sample_steps'],
            n_nodes=num_graphs,
            desc=desc,
        )

        # final sample (positions plus offset, and one-hot predictions)
        final = sample_chain[-1]  # (mu_pos_final, k_final, k_hat_final)
        pred_pos, one_hot = final[0] + offset[batch_ligand], final[1]

        pred_v = one_hot.argmax(dim=-1)  # predicted type indices per node
        pred_atom_type = [MAP_INDEX_TO_ATOM_TYPE_ONLY[i] for i in pred_v.tolist()]

        # create tensor of atom type indices (for any further tensor ops)
        atom_type = [MAP_ATOM_TYPE_ONLY_TO_INDEX[i] for i in pred_atom_type]
        atom_type = torch.tensor(atom_type, dtype=torch.long, device=x.device)

        out_data = {'h': pred_atom_type, 'x': pred_pos, 'batch': batch_ligand}
        return out_data

    def on_train_epoch_end(self) -> None:
        """Compute and log average epoch training loss."""
        if len(self.train_losses) == 0:
            epoch_loss = 0
        else:
            epoch_loss = torch.stack([x for x in self.train_losses]).mean()
        print(f"epoch_loss: {epoch_loss}")
        self.log("epoch_loss", epoch_loss, batch_size=self.cfg['train']['batch_size'])
        self.train_losses = []

    def configure_optimizers(self):
        """Instantiate optimizer and scheduler from configuration and return Lightning dict."""
        train_cfg = dict_to_namespace(self.cfg['train'])
        optimizer_cfg = dict_to_namespace(self.cfg['train']['optimizer'])
        self.optim = get_optimizer(optimizer_cfg, self)
        self.scheduler, self.get_last_lr = get_scheduler(train_cfg, self.optim)

        return {
            'optimizer': self.optim,
            'lr_scheduler': self.scheduler,
            # 'monitor': 'val_loss',  # optional
        }

    def test_step(self, batch, batch_idx):
        """Test step: deterministic encoding, sample multiple times and collect results."""
        original_x = batch['x']
        original_h = batch['h']
        original_batch = batch['batch']
        print(original_batch)

        # center positions for evaluation
        x, _ = center_pos(
            ligand_pos=original_x,
            batch_ligand=original_batch,
            mode=self.cfg['decoder_config']['center_pos_mode']
        )

        # map and encode deterministically
        h = [MAP_ATOM_TYPE_ONLY_TO_INDEX[i.item()] for i in original_h]
        h = torch.tensor(h, dtype=torch.long).to(x.device)
        K = self.cfg['encoder_config']['ligand_v_dim']
        one_hot_h = F.one_hot(h, K).float()  # [N, K]
        global_nodes, global_position, global_batch, _, _ = self.encode(one_hot_h, x, original_batch, deterministic=True)

        unique_batches = torch.unique(original_batch)
        original_unique_list = unique_batches.tolist()

        # prepare grouped initial structures
        grouped_results = []
        for batch_idx_val in unique_batches:
            mask = original_batch == batch_idx_val
            grouped_results.append({
                'initial': {
                    'x': original_x[mask].cpu().numpy().tolist(),
                    'h': original_h[mask].cpu().numpy().tolist(),
                    'batch': original_batch[mask].cpu().numpy().tolist(),
                },
                'samples': []
            })

        # save global positions for inspection
        self._save_global_position_as_xyz(global_position, original_batch, batch_idx)

        # draw multiple samples
        n_samples = self.cfg['evaluation']['num_samples']
        samples = [
            self.shared_sampling_step(
                batch,
                batch_idx,
                sample_num_atoms=self.cfg['evaluation']['sample_num_atoms'],
                desc=f'Test-{i}/{n_samples}',
                deterministic=True
            ) for i in range(n_samples)
        ]

        # collect per-molecule samples into grouped_results
        for sample_data in samples:
            sample_batch = sample_data['batch']
            for batch_idx_val in torch.unique(sample_batch):
                mask = sample_batch == batch_idx_val
                idx = original_unique_list.index(batch_idx_val.item())

                h_data = sample_data['h']
                if isinstance(h_data, torch.Tensor):
                    h_selected = h_data[mask].cpu().numpy().tolist()
                else:
                    mask_indices = torch.where(mask)[0].tolist()
                    h_selected = [h_data[i] for i in mask_indices]

                grouped_results[idx]['samples'].append({
                    'x': sample_data['x'][mask].cpu().numpy().tolist(),
                    'h': h_selected,
                    'batch': sample_batch[mask].cpu().numpy().tolist(),
                })

        if not hasattr(self, 'test_results') or self.test_results is None:
            self.test_results = []
        self.test_results.extend(grouped_results)
        return grouped_results

    def _save_global_position_as_xyz(self, global_position, batch, batch_idx):
        """Save global positions to an XYZ file for inspection (one file per call)."""
        print(f"global_position shape: {global_position.shape}")
        print(f"global_position: {global_position}")

        if len(global_position.shape) == 1:
            global_position = global_position.reshape(-1, 3)

        all_pos = global_position.cpu().numpy()

        timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        filename = os.path.join(self.save_dir, f'global_position_all_{timestamp}.xyz')

        with open(filename, 'w') as f:
            f.write(f"{len(all_pos)}\n")
            f.write("Global Position for All Batches\n")
            for (x, y, z) in all_pos:
                f.write(f"C {x} {y} {z}\n")

        print(f"Global position saved to: {filename}")

    def on_test_epoch_end(self):
        """Called after all test batches; save aggregated test results."""
        all_results = self.test_results

        processed_results = []
        for mol_group in all_results:
            processed_mol = {
                'initial': mol_group['initial'],
                'samples': mol_group['samples']
            }
            processed_results.append(processed_mol)

        self._save_results(processed_results)
        wandb.save(os.path.join(self.save_dir, f'test_results_*.json'))

    def _convert_tensor_to_dict(self, data_dict):
        """Helper: convert tensor-based batch dict into JSON-serializable python types."""
        return {
            'x': data_dict['x'].cpu().numpy().tolist(),
            'h': data_dict['h'],
            'batch': data_dict['batch'].cpu().numpy().tolist()
        }

    def _save_results(self, results):
        """Save final test results and configuration as a JSON file."""
        timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        save_path = os.path.join(self.save_dir, f'test_results_{timestamp}.json')

        output = {
            'config': self.cfg,
            'results': results,
            'save_time': timestamp,
            'num_molecules': len(results),
            'num_samples_per_mol': len(results[0]['samples']) if results else 0
        }

        with open(save_path, 'w') as f:
            json.dump(output, f, indent=2)

        print(f"save to: {save_path}")



class TrainLoopCharges(pl.LightningModule):
    def __init__(self, cfg, device=None):
        """Training loop wrapper

        Args:
            cfg (dict): Configuration dictionary (encoder/decoder/training/eval).
        """
        super().__init__()
        self.cfg = cfg

        # Instantiate encoder and decoder modules
        self.encoder = Encoder(**self.cfg['encoder_config'])
        decoder_config = dict(self.cfg['decoder_config_charge'])
        legacy_device = decoder_config.pop('device', None)
        selected_device = resolve_device(device if device is not None else legacy_device, self.cfg)
        self.decoder = BFN_charge(**decoder_config)

        # Linear layers used by the KL / latent parametrization
        self.Wh_mu = nn.Linear(
            self.cfg['encoder_config']['hidden_dim'],
            self.cfg['optimal_layer_config']['latent_dim']
        )
        self.Wh_log_var = nn.Linear(
            self.cfg['encoder_config']['hidden_dim'],
            self.cfg['optimal_layer_config']['latent_dim']
        )
        # Must be isotropic (scalar per node) to preserve equivariance
        self.Wx_log_var = nn.Linear(self.cfg['encoder_config']['hidden_dim'], 1)

        # bookkeeping / diagnostics
        self.time_records = np.zeros(6)
        self.encoder_config = self.cfg['encoder_config']
        self.decoder_config = self.cfg['decoder_config']

        self.train_losses = []
        self.log_time = False

        # print parameter counts for encoder / KL layer / decoder
        self.print_model_params()

        # directory for evaluation outputs
        self.save_dir = cfg['evaluation']['save_dir']
        os.makedirs(self.save_dir, exist_ok=True)

        self.test_result = []
        self.to(selected_device)

    def print_model_params(self):
        """Print number of trainable parameters"""
        def count_params(module):
            return sum(p.numel() for p in module.parameters() if p.requires_grad)

        encoder_params = count_params(self.encoder)
        optimal_layer_params = (
            count_params(self.Wh_log_var)
            + count_params(self.Wh_mu)
            + count_params(self.Wx_log_var)
        )
        decoder_params = count_params(self.decoder)
        charge_params = count_params(self.decoder.charge_head)
        total_params = encoder_params + decoder_params + optimal_layer_params

        print("\n" + "=" * 50)
        print(f"Encoder params: {encoder_params/1e6:.2f}M")
        print(f"KL layer params: {optimal_layer_params/1e6:.2f}M")
        print(f"Decoder params: {decoder_params/1e6:.2f}M")
        print(f"Of which {charge_params} are charge head params.")
        print(f"total params: {total_params/1e6:.2f}M")
        print("=" * 50 + "\n")

    def forward(self, x):
        """Forward is unused here. Kept as placeholder."""
        pass

    def encode(self, one_hot_h, x, batch_ligand, deterministic=False):
        """Encode input to latent representations and compute KL losses.

        Args:
            one_hot_h (Tensor): one-hot encoded ligand node types, shape [N, K].
            x (Tensor): node positions, shape [N, 3].
            batch_ligand (Tensor): batch index per node, shape [N].
            deterministic (bool): if True, do not sample from latent distributions.

        Returns:
            Zh_sampled (Tensor): sampled / deterministic latent for node attributes.
            Zx_sampled (Tensor): sampled / deterministic latent for node positions.
            global_batch (Tensor): batch indices corresponding to global nodes.
            Zh_kl_loss (Tensor): KL loss for Zh (attribute latent).
            Zx_kl_loss (Tensor): KL loss for Zx (position latent).
        """
        # encoder returns global node features, global positions, and batch mapping
        global_h, global_x, global_batch = self.encoder(one_hot_h, x, batch_ligand)

        # Parameterize latent Gaussian for Zh (attribute latent)
        Zh_mu = self.Wh_mu(global_h)
        Zh_log_var = -torch.abs(self.Wh_log_var(global_h))  # ensure non-positive values

        # For Zx, use global_x as mean and predict a scalar log-variance per global node
        Zx_mu = global_x.clone()

        # clamp log variance to avoid excessively large variance
        upper = torch.log(
            torch.tensor(self.cfg['train']['kl_loss']['sigma2']**2,
                         device=Zx_mu.device, dtype=Zx_mu.dtype)
        )
        raw = self.Wx_log_var(global_h).expand_as(Zx_mu)
        Zx_log_var = torch.clamp(raw, max=upper)

        # number of graphs in the batch (unique batch indices)
        data_size = torch.unique(global_batch).size(0)

        # KL divergences (averaged per-element and per-dimension)
        Zh_kl_loss = -0.5 * torch.sum(
            1.0 + Zh_log_var - Zh_mu * Zh_mu - torch.exp(Zh_log_var)
        ) / (data_size * Zh_mu.shape[-1])

        Zx_kl_loss = -0.5 * torch.sum(
            1.0 + Zx_log_var - (Zx_mu * Zx_mu + torch.exp(Zx_log_var)) / (self.cfg['train']['kl_loss']['sigma2'])**2
        ) / (data_size * Zx_mu.shape[-1])

        # Reparameterization: sample if not deterministic
        Zh_sampled = Zh_mu if deterministic else Zh_mu + torch.exp(Zh_log_var / 2) * torch.randn_like(Zh_mu)
        Zx_sampled = Zx_mu if deterministic else Zx_mu + torch.exp(Zx_log_var / 2) * torch.randn_like(Zx_mu)

        return Zh_sampled, Zx_sampled, global_batch, Zh_kl_loss, Zx_kl_loss

    def training_step(self, batch, batch_idx):
        """Single training step: encode, compute decoder losses, log and return loss."""
        t1 = time()

        h = batch['h']           # original atom types [N_node, 1]
        x = batch['x']           # positions [N_node, 3]
        ligand_charges = batch['charges'] 
        batch_ligand = batch['batch']  # batch indices per node [N_node]

        t2 = time()
        # number of graphs in this batch
        num_graphs = batch_ligand.max().item() + 1

        # center ligand positions according to configured mode
        x, _ = center_pos(
            ligand_pos=x,
            batch_ligand=batch_ligand,
            mode=self.cfg['decoder_config']['center_pos_mode']
        )

        t3 = time()

        # sample a time t for each graph, then select per-node by batch index
        t = torch.rand([num_graphs, 1], dtype=x.dtype, device=x.device).index_select(0, batch_ligand)

        # optional clamping if decoder uses continuous t
        if not self.cfg['decoder_config']['use_discrete_t'] and not self.cfg['decoder_config']['destination_prediction']:
            t = torch.clamp(t, min=self.decoder.t_min)

        t4 = time()

        # map atom types to indices and one-hot encode
        h = [MAP_ATOM_TYPE_ONLY_TO_INDEX[i.item()] for i in h]
        h = torch.tensor(h, dtype=torch.long).to(x.device)
        K = self.cfg['encoder_config']['ligand_v_dim']
        one_hot_h = F.one_hot(h, K).float()  # [N, K]

        # encode to obtain latent samples and KL losses
        Zh_sampled, Zx_sampled, global_batch, Zh_kl_loss, Zx_kl_loss = self.encode(
            one_hot_h, x, batch_ligand, deterministic=False
        )

        # decoder loss for this step (returns c_loss, d_loss, discretised_loss)
        c_loss, d_loss, discretised_loss = self.decoder.loss_one_step(
            t,
            protein_pos=Zx_sampled,
            protein_v=Zh_sampled,
            batch_protein=global_batch,
            ligand_pos=x,
            ligand_v=h,
            batch_ligand=batch_ligand,
            ligand_charges=ligand_charges
        )

        # reconstruction and KL combined loss (mean over batch)
        recon_loss = torch.mean(
            self.cfg['train']['recon_loss']['c_loss_weight'] * c_loss
            + self.cfg['train']['recon_loss']['d_loss_weight'] * d_loss
            + discretised_loss
        )
        kl_loss = torch.mean(
            self.cfg['train']['kl_loss']['Zh_kl_loss_weight'] * Zh_kl_loss
            + self.cfg['train']['kl_loss']['Zx_kl_loss_weight'] * Zx_kl_loss
        )
        loss = (
            self.cfg['train']['recon_loss']['recon_loss_weight'] * recon_loss
            + self.cfg['train']['kl_loss']['kl_loss_weight'] * kl_loss
        )

        # logging to W&B and Lightning
        wandb.log({
            'lr': self.get_last_lr(),
            'train_loss': loss.item(),
            'train_recon_loss': recon_loss.item(),
            'train_kl_loss': kl_loss.item()
        })

        t5 = time()

        self.log_dict(
            {
                'lr': self.get_last_lr(),
                'train_loss': loss.item(),
                'recon_loss': recon_loss.item(),
                'kl_loss': kl_loss.item()
            },
            on_step=True,
            prog_bar=True,
            batch_size=self.cfg['train']['batch_size'],
        )

        # skip updates when loss is not finite
        if not torch.isfinite(loss):
            return None

        self.train_losses.append(loss.clone().detach().cpu())

        t0 = time()

        # optional timing diagnostics
        if self.log_time:
            self.time_records = np.vstack((self.time_records, [t0, t1, t2, t3, t4, t5]))
            print(f'step total time: {self.time_records[-1, 0] - self.time_records[-1, 1]}, batch size: {num_graphs}')
            print(f'\tpl call & data access: {self.time_records[-1, 1] - self.time_records[-2, 0]}')
            print(f'\tunwrap data: {self.time_records[-1, 2] - self.time_records[-1, 1]}')
            print(f'\tadd noise & center pos: {self.time_records[-1, 3] - self.time_records[-1, 2]}')
            print(f'\tsample t: {self.time_records[-1, 4] - self.time_records[-1, 3]}')
            print(f'\tget loss: {self.time_records[-1, 5] - self.time_records[-1, 4]}')
            print(f'\tlogging: {self.time_records[-1, 0] - self.time_records[-1, 5]}')

        return loss

    def validation_step(self, batch, batch_idx):
        """Validation: deterministic encode, compute losses, sample and compute similarity metrics."""
        original_h = batch['h']
        original_x = batch['x']
        original_ligand_charges = batch['charges'] 
        
        batch_ligand = batch['batch']

        # center ligand positions for evaluation
        x, _ = center_pos(
            ligand_pos=original_x,
            batch_ligand=batch_ligand,
            mode=self.cfg['decoder_config']['center_pos_mode']
        )

        num_graphs = batch_ligand.max().item() + 1

        # sample time t per graph, clamp if needed
        t = torch.rand([num_graphs, 1], dtype=x.dtype, device=x.device).index_select(0, batch_ligand)
        if not self.cfg['decoder_config']['use_discrete_t'] and not self.cfg['decoder_config']['destination_prediction']:
            t = torch.clamp(t, min=self.decoder.t_min)

        # map atom types and one-hot encode
        h = [MAP_ATOM_TYPE_ONLY_TO_INDEX[i.item()] for i in original_h]
        h = torch.tensor(h, dtype=torch.long).to(x.device)
        K = self.cfg['encoder_config']['ligand_v_dim']
        one_hot_h = F.one_hot(h, K).float()  # [N, K]

        # deterministic encode for validation
        Zh_sampled, Zx_sampled, global_batch, Zh_kl_loss, Zx_kl_loss = self.encode(
            one_hot_h, x, batch_ligand, deterministic=True
        )

        # compute decoder losses
        c_loss, d_loss, discretised_loss = self.decoder.loss_one_step(
            t,
            protein_pos=Zx_sampled,
            protein_v=Zh_sampled,
            batch_protein=global_batch,
            ligand_pos=x,
            ligand_v=h,
            batch_ligand=batch_ligand,
            ligand_charges=original_ligand_charges
        )

        recon_loss = torch.mean(
            self.cfg['train']['recon_loss']['c_loss_weight'] * c_loss
            + self.cfg['train']['recon_loss']['d_loss_weight'] * d_loss
            + discretised_loss
        )
        kl_loss = torch.mean(
            self.cfg['train']['kl_loss']['Zh_kl_loss_weight'] * Zh_kl_loss
            + self.cfg['train']['kl_loss']['Zx_kl_loss_weight'] * Zx_kl_loss
        )
        loss = recon_loss + kl_loss

        # generate deterministic samples for similarity evaluation
        generated_data = self.shared_sampling_step(batch, batch_idx, sample_num_atoms='ref', desc='Val', deterministic=True)

        molecule_builder = MoleculeBuilder()

        generated_h = torch.tensor(generated_data['h'], dtype=torch.long).to(x.device)
        generated_x = generated_data['x'].to(x.device)
        generated_batch = generated_data['batch'].to(x.device)

        unique_batches = torch.unique(batch_ligand)

        similarities = []
        for batch_idx_val in unique_batches:
            original_mask = batch_ligand == batch_idx_val
            original_atoms = original_h[original_mask].view(-1).cpu().tolist()
            original_atoms = [int(a) for a in original_atoms]  # Avoid a nested list
            original_coords = original_x[original_mask].cpu().numpy()

            generated_mask = generated_batch == batch_idx_val
            generated_atoms = generated_h[generated_mask].view(-1).cpu().tolist()
            generated_atoms = [int(a) for a in generated_atoms]
            generated_coords = generated_x[generated_mask].cpu().numpy()

            original_mol = molecule_builder.build_mol(original_coords, original_atoms)
            generated_mol = molecule_builder.build_mol(generated_coords, generated_atoms)

            if original_mol is None or generated_mol is None:
                print(f"Warning: Invalid molecule for batch_idx_val {batch_idx_val.item()}")
                continue

            similarity = molecule_builder.compute_iou(original_mol, generated_mol)
            similarities.append(similarity)

        avg_similarity = torch.tensor(similarities).mean().item() if similarities else 0.0

        # log metrics
        wandb.log({
            'val_loss': loss.item(),
            'val_recon_loss': recon_loss.item(),
            'val_kl_loss': kl_loss.item(),
            'val_similarity': avg_similarity,
        })
        self.log_dict({
            'val_loss': loss.item(),
            'val_similarity': avg_similarity,
        },
            prog_bar=True,
            logger=True,
            on_step=True,
            sync_dist=True,
            batch_size=self.cfg['evaluation']['batch_size'],
        )

        return loss

    def shared_sampling_step(self, batch, batch_idx, sample_num_atoms, desc='', deterministic=True):
        """Shared sampling routine used in validation and testing.

        Args:
            batch (dict): input batch with 'h', 'x', 'batch'
            sample_num_atoms (str): 'prior' or 'ref' (or other modes raise ValueError)
            desc (str): description passed to decoder.sample for logging
            deterministic (bool): whether to use deterministic latent samples

        Returns:
            out_data (dict): {'h': predicted atom types (list), 'x': positions Tensor, 'batch': batch Tensor}
        """
        h = batch['h']
        x = batch['x']
        batch_ligand = batch['batch']

        num_graphs = batch_ligand.max().item() + 1  # number of molecules in this batch
        n_nodes = batch_ligand.size(0)  # total nodes

        # center positions and get offset for restoring coordinates later
        x, offset = center_pos(ligand_pos=x, batch_ligand=batch_ligand, mode=True)

        # decide per-graph atom counts
        if sample_num_atoms == 'prior':
            ligand_num_atoms = []
            for data_id in range(len(batch)):
                data = batch[data_id]
                pocket_size = atom_num.get_space_size(data.protein_pos.detach().cpu().numpy() * self.cfg['data']['normalizer_dict']['pos'])
                ligand_num_atoms.append(atom_num.sample_atom_num(pocket_size).astype(int))
            batch_ligand = torch.repeat_interleave(torch.arange(len(batch)), torch.tensor(ligand_num_atoms)).to(x.device)
            ligand_num_atoms = torch.tensor(ligand_num_atoms, dtype=torch.long, device=x.device)
        elif sample_num_atoms == 'ref':
            batch_ligand = batch_ligand
            ligand_num_atoms = scatter_sum(torch.ones_like(batch_ligand), batch_ligand, dim=0).to(x.device)
        else:
            raise ValueError(f"sample_num_atoms mode: {sample_num_atoms} not supported")

        # cumulative counts (useful for indexing if needed)
        ligand_cum_atoms = torch.cat([
            torch.tensor([0], dtype=torch.long, device=x.device),
            ligand_num_atoms.cumsum(dim=0)
        ])

        # map atom types to indices and one-hot encode
        h = [MAP_ATOM_TYPE_ONLY_TO_INDEX[i.item()] for i in h]
        h = torch.tensor(h, dtype=torch.long).to(x.device)
        K = self.cfg['encoder_config']['ligand_v_dim']
        h = F.one_hot(h, K).float()  # [N, K]

        # encode (deterministic or sampled)
        global_nodes, global_position, global_batch, Zh_kl_loss, Zx_kl_loss = self.encode(h, x, batch_ligand, deterministic=deterministic)

        # sample molecule chain from decoder
        theta_chain, sample_chain, y_chain = self.decoder.sample(
            protein_pos=global_position,
            protein_v=global_nodes,
            batch_protein=global_batch,
            batch_ligand=batch_ligand,
            sample_steps=self.cfg['evaluation']['sample_steps'],
            n_nodes=num_graphs,
            desc=desc,
        )

        # final sample (positions plus offset, and one-hot predictions)
        final = sample_chain[-1]  # (mu_pos_final, k_final, k_hat_final)
        pred_pos, one_hot = final[0] + offset[batch_ligand], final[1]

        pred_v = one_hot.argmax(dim=-1)  # predicted type indices per node
        pred_atom_type = [MAP_INDEX_TO_ATOM_TYPE_ONLY[i] for i in pred_v.tolist()]

        # create tensor of atom type indices (for any further tensor ops)
        atom_type = [MAP_ATOM_TYPE_ONLY_TO_INDEX[i] for i in pred_atom_type]
        atom_type = torch.tensor(atom_type, dtype=torch.long, device=x.device)

        out_data = {'h': pred_atom_type, 'x': pred_pos, 'batch': batch_ligand}
        return out_data

    def on_train_epoch_end(self) -> None:
        """Compute and log average epoch training loss."""
        if len(self.train_losses) == 0:
            epoch_loss = 0
        else:
            epoch_loss = torch.stack([x for x in self.train_losses]).mean()
        print(f"epoch_loss: {epoch_loss}")
        self.log("epoch_loss", epoch_loss, batch_size=self.cfg['train']['batch_size'])
        self.train_losses = []

    def configure_optimizers(self):
        """Instantiate optimizer and scheduler from configuration and return Lightning dict."""
        train_cfg = dict_to_namespace(self.cfg['train'])
        optimizer_cfg = dict_to_namespace(self.cfg['train']['optimizer'])
        self.optim = get_optimizer(optimizer_cfg, self)
        self.scheduler, self.get_last_lr = get_scheduler(train_cfg, self.optim)

        return {
            'optimizer': self.optim,
            'lr_scheduler': self.scheduler,
            # 'monitor': 'val_loss',  # optional
        }

    def test_step(self, batch, batch_idx):
        """Test step: deterministic encoding, sample multiple times and collect results."""
        original_x = batch['x']
        original_h = batch['h']
        original_batch = batch['batch']
        print(original_batch)

        # center positions for evaluation
        x, _ = center_pos(
            ligand_pos=original_x,
            batch_ligand=original_batch,
            mode=self.cfg['decoder_config']['center_pos_mode']
        )

        # map and encode deterministically
        h = [MAP_ATOM_TYPE_ONLY_TO_INDEX[i.item()] for i in original_h]
        h = torch.tensor(h, dtype=torch.long).to(x.device)
        K = self.cfg['encoder_config']['ligand_v_dim']
        one_hot_h = F.one_hot(h, K).float()  # [N, K]
        global_nodes, global_position, global_batch, _, _ = self.encode(one_hot_h, x, original_batch, deterministic=True)

        unique_batches = torch.unique(original_batch)
        original_unique_list = unique_batches.tolist()

        # prepare grouped initial structures
        grouped_results = []
        for batch_idx_val in unique_batches:
            mask = original_batch == batch_idx_val
            grouped_results.append({
                'initial': {
                    'x': original_x[mask].cpu().numpy().tolist(),
                    'h': original_h[mask].cpu().numpy().tolist(),
                    'batch': original_batch[mask].cpu().numpy().tolist(),
                },
                'samples': []
            })

        # save global positions for inspection
        self._save_global_position_as_xyz(global_position, original_batch, batch_idx)

        # draw multiple samples
        n_samples = self.cfg['evaluation']['num_samples']
        samples = [
            self.shared_sampling_step(
                batch,
                batch_idx,
                sample_num_atoms=self.cfg['evaluation']['sample_num_atoms'],
                desc=f'Test-{i}/{n_samples}',
                deterministic=True
            ) for i in range(n_samples)
        ]

        # collect per-molecule samples into grouped_results
        for sample_data in samples:
            sample_batch = sample_data['batch']
            for batch_idx_val in torch.unique(sample_batch):
                mask = sample_batch == batch_idx_val
                idx = original_unique_list.index(batch_idx_val.item())

                h_data = sample_data['h']
                if isinstance(h_data, torch.Tensor):
                    h_selected = h_data[mask].cpu().numpy().tolist()
                else:
                    mask_indices = torch.where(mask)[0].tolist()
                    h_selected = [h_data[i] for i in mask_indices]

                grouped_results[idx]['samples'].append({
                    'x': sample_data['x'][mask].cpu().numpy().tolist(),
                    'h': h_selected,
                    'batch': sample_batch[mask].cpu().numpy().tolist(),
                })

        if not hasattr(self, 'test_results') or self.test_results is None:
            self.test_results = []
        self.test_results.extend(grouped_results)
        return grouped_results

    def _save_global_position_as_xyz(self, global_position, batch, batch_idx):
        """Save global positions to an XYZ file for inspection (one file per call)."""
        print(f"global_position shape: {global_position.shape}")
        print(f"global_position: {global_position}")

        if len(global_position.shape) == 1:
            global_position = global_position.reshape(-1, 3)

        all_pos = global_position.cpu().numpy()

        timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        filename = os.path.join(self.save_dir, f'global_position_all_{timestamp}.xyz')

        with open(filename, 'w') as f:
            f.write(f"{len(all_pos)}\n")
            f.write("Global Position for All Batches\n")
            for (x, y, z) in all_pos:
                f.write(f"C {x} {y} {z}\n")

        print(f"Global position saved to: {filename}")

    def on_test_epoch_end(self):
        """Called after all test batches; save aggregated test results."""
        all_results = self.test_results

        processed_results = []
        for mol_group in all_results:
            processed_mol = {
                'initial': mol_group['initial'],
                'samples': mol_group['samples']
            }
            processed_results.append(processed_mol)

        self._save_results(processed_results)
        wandb.save(os.path.join(self.save_dir, f'test_results_*.json'))

    def _convert_tensor_to_dict(self, data_dict):
        """Helper: convert tensor-based batch dict into JSON-serializable python types."""
        return {
            'x': data_dict['x'].cpu().numpy().tolist(),
            'h': data_dict['h'],
            'batch': data_dict['batch'].cpu().numpy().tolist()
        }

    def _save_results(self, results):
        """Save final test results and configuration as a JSON file."""
        timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        save_path = os.path.join(self.save_dir, f'test_results_{timestamp}.json')

        output = {
            'config': self.cfg,
            'results': results,
            'save_time': timestamp,
            'num_molecules': len(results),
            'num_samples_per_mol': len(results[0]['samples']) if results else 0
        }

        with open(save_path, 'w') as f:
            json.dump(output, f, indent=2)

        print(f"save to: {save_path}")
