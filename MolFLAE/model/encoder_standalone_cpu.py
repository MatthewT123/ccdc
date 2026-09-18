import torch
import torch.nn as nn
import torch.nn.functional as F
from model.uni_transformer import UniTransformerO2TwoUpdateGeneral
from model.train_loop import center_pos

from utils.config import load_config

# load the config
cfg = load_config("config.yaml")

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


class Encoder(nn.Module):
    def __init__(self, num_blocks, num_layers, hidden_dim, n_heads=1, knn=32,
                 num_r_gaussian=20, edge_feat_dim=4, num_node_types=8, act_fn='relu',
                 norm=True, cutoff_mode='global', ew_net_type='r', num_init_x2h=1,
                 num_init_h2x=0, num_x2h=1, num_h2x=1, r_max=10., x2h_out_fc=True,
                 sync_twoup=False, global_node_num=10, ligand_v_dim=9):
        super().__init__()

        self.hidden_dim = hidden_dim  # number of virtual nodes (H)
        self.ligand_v_dim = ligand_v_dim  # number of atomic features (K)
        self.global_node_num = global_node_num

        self.input_adaptor = MLP(
            in_dim=ligand_v_dim,
            out_dim=hidden_dim,
            hidden_dim=hidden_dim,
            num_layer=2,
            norm=True,
            act_fn='relu',
            act_last=False)
        self.unet = UniTransformerO2TwoUpdateGeneral(
            num_blocks=num_blocks,
            num_layers=num_layers,
            hidden_dim=hidden_dim,
            n_heads=n_heads,
            knn=knn,
            num_r_gaussian=num_r_gaussian,
            edge_feat_dim=edge_feat_dim,
            num_node_types=num_node_types,
            act_fn=act_fn,
            norm=norm,
            cutoff_mode=cutoff_mode,
            ew_net_type=ew_net_type,
            num_init_x2h=num_init_x2h,
            num_init_h2x=num_init_h2x,
            num_x2h=num_x2h,
            num_h2x=num_h2x,
            r_max=r_max,
            x2h_out_fc=x2h_out_fc,
            sync_twoup=sync_twoup
        )
        # learnable global_h layer
        self.global_h = nn.Parameter(torch.randn(self.global_node_num, self.hidden_dim))
        nn.init.normal_(self.global_h, mean=0, std=0.1)

    def __repr__(self):
        return (
            f"Encoder:\n{self.unet}\n"
        )

    def forward(self, h, x, batch):
        """Forward method

        h (Tensor): one-hot encoded ligand node types, shape [N, K]
        x (Tensor): ligand node positions, shape [N, 3]
        batch (Tensor): batch index per node, shape N
        """

        # Add global nodes
        batch_size = batch.max().item() + 1  # number of molecules
        global_h = self.global_h.repeat(batch_size, 1)
        global_x = torch.zeros(
            (batch_size * self.global_node_num, 3),
            device=x.device
        )
        global_batch = torch.repeat_interleave(
            torch.arange(batch_size, device=batch.device),
            repeats=self.global_node_num
        )

        # Concat normal and global nodes
        x_new = torch.cat([x, global_x], dim=0)
        batch_new = torch.cat([batch, global_batch], dim=0)
        h = self.input_adaptor(h)  # ligand_v_dim → hidden_dim
        h_new = torch.cat([h, global_h], dim=0)

        # generate mask: normal_nodes=0, global_nodes=1
        mask_ligand_new = torch.cat([
            torch.zeros(len(h), dtype=torch.bool, device=h.device),
            torch.ones(batch_size * self.global_node_num, dtype=torch.bool, device=h.device)
        ], dim=0)

        # Update h and x using the network
        outputs = self.unet(h_new, x_new, mask_ligand_new, batch_new, return_edge=True)

        # Extract global nodes
        h_updated = outputs['h']
        x_updated = outputs['x']
        global_nodes_updated = h_updated[-batch_size * self.global_node_num:]
        global_positions_updated = x_updated[-batch_size * self.global_node_num:]

        return global_nodes_updated, global_positions_updated, global_batch

    def encode(self, one_hot_h, x, batch_ligand, deterministic=False):
        # global_batch：[0,0,……(10),1,1,……(10),…………]
        # global_h：hidden_dim
        global_h, global_x, global_batch = self.forward(
            one_hot_h,
            x,
            batch_ligand
        )

        # mu&var;global_h:hidden_dim→latent_dim
        Zh_mu = self.Wh_mu(global_h)
        Zh_log_var = -torch.abs(self.Wh_log_var(global_h))
        Zx_mu = global_x.clone()

        # clamp log_var to avoid too large variance
        upper = torch.log(torch.tensor(cfg['train']['kl_loss']['sigma2']**2, device=Zx_mu.device, dtype=Zx_mu.dtype))
        raw = self.Wx_log_var(global_h).expand_as(Zx_mu)
        Zx_log_var = torch.clamp(raw, max=upper)

        data_size = torch.unique(global_batch).size(0)

        Zh_kl_loss = -0.5 * torch.sum(1.0 + Zh_log_var - Zh_mu * Zh_mu - torch.exp(Zh_log_var)) / (data_size * Zh_mu.shape[-1])
        Zx_kl_loss = -0.5 * torch.sum(1.0 + Zx_log_var -
                                      (Zx_mu * Zx_mu + torch.exp(Zx_log_var)) / (cfg['train']['kl_loss']['sigma2'])**2) / (data_size * Zx_mu.shape[-1])

        # rsample
        Zh_sampled = Zh_mu if deterministic else Zh_mu + torch.exp(Zh_log_var / 2) * torch.randn_like(Zh_mu)
        Zx_sampled = Zx_mu if deterministic else Zx_mu + torch.exp(Zx_log_var / 2) * torch.randn_like(Zx_mu)

        return Zh_sampled, Zx_sampled, global_batch, Zh_kl_loss, Zx_kl_loss


class MLP(nn.Module):
    """MLP with the same hidden dim across all layers."""

    def __init__(self, in_dim, out_dim, hidden_dim, num_layer=2, norm=True, act_fn='relu', act_last=False):
        super().__init__()
        layers = []
        for layer_idx in range(num_layer):
            if layer_idx == 0:
                layers.append(nn.Linear(in_dim, hidden_dim))
            elif layer_idx == num_layer - 1:
                layers.append(nn.Linear(hidden_dim, out_dim))
            else:
                layers.append(nn.Linear(hidden_dim, hidden_dim))
            if layer_idx < num_layer - 1 or act_last:
                if norm:
                    layers.append(nn.LayerNorm(hidden_dim))
                layers.append(NONLINEARITIES[act_fn])
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class Swish(nn.Module):
    def __init__(self):
        super(Swish, self).__init__()
        self.beta = nn.Parameter(torch.tensor(1.0))

    def forward(self, x):
        return x * torch.sigmoid(self.beta * x)


NONLINEARITIES = {
    "tanh": nn.Tanh(),
    "relu": nn.ReLU(),
    "softplus": nn.Softplus(),
    "elu": nn.ELU(),
    "swish": Swish(),
    'silu': nn.SiLU()
}


def molecule_to_latent(encoder, mol_entry, return_numpy=False):
    """
    Compute latent representation (Zh, Zx) for a single molecule entry from mol_1000.

    Args:
        encoder: the trained encoder component of the model with an `encode(one_hot_h, x, batch_ligand, deterministic=True)` method.
               Model should be in eval() mode (this function will not call model.eval() automatically).
               The function will detect device from model parameters (next(model.parameters()).device).
        mol_entry: dict-like containing at least:
                   - 'h': tensor of atom types (integers or torch tensors)
                   - 'x': tensor of shape (N, 3) with coordinates
                   - 'atom_num': integer (optional)
        return_numpy: if True, returns numpy arrays (Zh_cpu, Zx_cpu, global_batch_cpu). Default False.

    Returns:
        Zh, Zx, global_batch  (either torch tensors on model device or numpy arrays if return_numpy=True)
    """
    # load the config
    cfg = load_config("config.yaml")

    # The caller configures placement through encoder.to(device).
    device = next(encoder.parameters()).device

    # pull data
    h = mol_entry['h']  # expect 1D tensor-like of atom types
    x = mol_entry['x']  # expect shape (N, 3)

    # Ensure tensors and move to device
    x = x.to(device)
    # Convert h to torch.LongTensor on device. If h is already tensor, make sure it's long.
    if not isinstance(h, torch.Tensor):
        h = torch.tensor(h, dtype=torch.long, device=device)
    else:
        h = h.to(device).long()

    # Chech if atom types should be mapped to indices
    needs_mapping = False
    K = cfg['encoder_config']['ligand_v_dim']
    if h.max().item() >= K:  # likely not indices
        needs_mapping = True

    if needs_mapping:
        h = torch.tensor([MAP_ATOM_TYPE_ONLY_TO_INDEX[int(i.item())] for i in h], dtype=torch.long, device=device)

    # One-hot encode using encoder ligand dim K
    one_hot_h = F.one_hot(h, num_classes=K).float().to(device)

    # Build a batch vector for a single molecule: all zeros
    batch_ligand = torch.zeros_like(h, dtype=torch.long, device=device)

    # Center positions
    x_centered, _ = center_pos(x, batch_ligand, mode=True)

    # Encode into latent space
    encoder.eval()
    with torch.no_grad():
        Zh, Zx, global_batch, Zh_kl_loss, Zx_kl_loss = encoder.encode(one_hot_h, x_centered, batch_ligand, deterministic=True)

    if return_numpy:
        Zh_cpu = Zh.cpu().numpy()
        Zx_cpu = Zx.cpu().numpy()
        global_batch_cpu = global_batch.cpu().numpy()
        return Zh_cpu, Zx_cpu, global_batch_cpu

    return Zh, Zx, global_batch
