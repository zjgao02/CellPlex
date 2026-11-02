"""CELLO model for cell-level protein prediction from H&E.

Adapted from CELLO (github.com/zjgao02/CELLO, cello/model.py), with the encoders used in CellPlex (UNI2-h,
Virchow2, H-optimus-0); "genes" are the proteins of a cohort's panel here.

One pathology-foundation-model (PFM) forward pass per 224x224 tile produces a 2-D token map. Each cell is queried at
its (sub-token) location by bilinear grid sampling, refined by a distance-decay cross-attention with 2-D RoPE over
the tile's tokens, normalised and fused with the tile CLS token, and decoded by a two-layer MLP head. An auxiliary
head on the CLS token predicts the summed levels of all cells in the tile.
"""

import math
from typing import Dict, List, Optional

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.data import resolve_data_config


# --------------------------------------------------------------------------
# Vision encoders
# --------------------------------------------------------------------------
# ``timm_kwargs`` are the model-card arguments of each backbone; ``patch_size`` defines the token grid
# (224 / patch_size per side).
ENCODER_REGISTRY = {
    "uni2": {
        "hub_id": "hf-hub:MahmoodLab/UNI2-h",
        "timm_kwargs": {"img_size": 224, "patch_size": 14, "depth": 24, "num_heads": 24, "init_values": 1e-5,
                        "embed_dim": 1536, "mlp_ratio": 2.66667 * 2, "no_embed_class": True,
                        "mlp_layer": timm.layers.SwiGLUPacked, "act_layer": nn.SiLU, "reg_tokens": 8,
                        "dynamic_img_size": True},
        "patch_size": 14,
    },
    "virchow2": {
        "hub_id": "hf-hub:paige-ai/Virchow2",
        "timm_kwargs": {"mlp_layer": timm.layers.SwiGLUPacked, "act_layer": nn.SiLU},
        "patch_size": 14,
    },
    "hoptimus0": {
        "hub_id": "hf-hub:bioptimus/H-optimus-0",
        "timm_kwargs": {"init_values": 1e-5, "dynamic_img_size": False},
        "patch_size": 14,
    },
}


def build_encoder(name: str, pretrained: bool = True) -> nn.Module:
    if name not in ENCODER_REGISTRY:
        raise ValueError(f"Unknown vision_encoder '{name}'. Choices: {list(ENCODER_REGISTRY)}")
    cfg = ENCODER_REGISTRY[name]
    return timm.create_model(cfg["hub_id"], pretrained=pretrained, num_classes=0,
                             **cfg["timm_kwargs"])


# --------------------------------------------------------------------------
# 2-D rotary position embedding
# --------------------------------------------------------------------------
def apply_rope_2d(q, k, pos_q, pos_k):
    """Rotate queries and keys with a 2-D RoPE.

    q: (N, h, dh)   k: (N, h, Ni, dh)   pos_q: (2,)   pos_k: (Ni, 2)
    The first half of each head encodes x, the second half encodes y.
    """
    device = q.device
    dh = q.shape[-1]
    half, quarter = dh // 2, dh // 4
    inv_freq = 1.0 / (10000 ** (torch.arange(quarter, device=device).float() / quarter))

    def _rot(vec, pos_scalar):
        s = torch.sin(pos_scalar * inv_freq)
        c = torch.cos(pos_scalar * inv_freq)
        return torch.stack([vec[..., 0] * c - vec[..., 1] * s,
                            vec[..., 0] * s + vec[..., 1] * c], dim=-1)

    qx = q[..., :half].view(*q.shape[:-1], quarter, 2)
    qy = q[..., half:].view(*q.shape[:-1], quarter, 2)
    qx = _rot(qx, pos_q[0])
    qy = _rot(qy, pos_q[1])
    q = torch.cat([qx.reshape(*q.shape[:-1], half),
                   qy.reshape(*q.shape[:-1], half)], dim=-1)

    kx = k[..., :half].view(*k.shape[:-1], quarter, 2)
    ky = k[..., half:].view(*k.shape[:-1], quarter, 2)
    pos_kx = pos_k[:, 0].to(device).unsqueeze(0).unsqueeze(0).unsqueeze(-1)
    pos_ky = pos_k[:, 1].to(device).unsqueeze(0).unsqueeze(0).unsqueeze(-1)
    kx = _rot(kx, pos_kx)
    ky = _rot(ky, pos_ky)
    k = torch.cat([kx.reshape(*k.shape[:-1], half),
                   ky.reshape(*k.shape[:-1], half)], dim=-1)
    return q, k


# --------------------------------------------------------------------------
# Sub-modules
# --------------------------------------------------------------------------
class GeneHead(nn.Module):
    """Two-layer MLP with a non-negative (ReLU) output."""

    def __init__(self, d=768, hidden=512, out_dim=0):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(d, hidden), nn.GELU(), nn.Dropout(0.2),
            nn.Linear(hidden, out_dim))

    def forward(self, z):
        return F.relu(self.fc(z))


class CrossAttnRoPE(nn.Module):
    """Single-query multi-head cross-attention with 2-D RoPE and an additive bias."""

    def __init__(self, d=768, nhead=8, dropout=0.2):
        super().__init__()
        self.d, self.h, self.dh = d, nhead, d // nhead
        self.q = nn.Linear(d, d)
        self.k = nn.Linear(d, d)
        self.v = nn.Linear(d, d)
        self.o = nn.Linear(d, d)
        self.dropout = nn.Dropout(dropout)

    def forward(self, q_vec, kv_img, pos_img_xy, bias_img=None):
        N, Ni, _ = kv_img.shape
        q = self.q(q_vec).view(N, self.h, self.dh)
        k = self.k(kv_img).view(N, Ni, self.h, self.dh).transpose(1, 2)
        v = self.v(kv_img).view(N, Ni, self.h, self.dh).transpose(1, 2)

        # The query is rotated with the mean token-grid position, i.e. the same
        # constant position for every cell; cell location enters the attention
        # through the grid-sampled query and the distance-decay bias.
        pos_q = pos_img_xy.float().mean(0).to(q.device)
        q, k = apply_rope_2d(q, k, pos_q, pos_img_xy.to(q.device))

        attn = (q.unsqueeze(2) * k).sum(-1) / math.sqrt(self.dh)
        if bias_img is not None:
            attn = attn + bias_img.unsqueeze(1)
        attn = self.dropout(F.softmax(attn, dim=-1))
        ctx = (attn.unsqueeze(-1) * v).sum(2)
        return self.o(ctx.reshape(N, self.d))


# --------------------------------------------------------------------------
# CELLO
# --------------------------------------------------------------------------
class CELLO(nn.Module):
    def __init__(self, union_genes: List[str], config: Optional[dict] = None,
                 pretrained_encoder: bool = True):
        super().__init__()
        self.config = dict(config or {})
        self.d = self.config.get("model_dim", 768)
        self.heads = self.config.get("heads", 8)
        self.drop = self.config.get("dropout", 0.2)
        self.img_size = self.config.get("img_size", 224)
        self.use_distance_bias = self.config.get("use_distance_bias", True)
        self.cell_feature_mode = self.config.get("cell_feature_mode", "cross_attn")
        if self.cell_feature_mode not in ("cross_attn", "grid"):
            raise ValueError(f"cell_feature_mode must be 'cross_attn' or 'grid', "
                             f"got '{self.cell_feature_mode}'")

        # ---- backbone ----
        self.vision_encoder = self.config.get("vision_encoder", "virchow2")
        self.encoder = build_encoder(self.vision_encoder, pretrained=pretrained_encoder)
        self.backbone_dim = self.encoder.embed_dim
        self.patch_size = ENCODER_REGISTRY[self.vision_encoder]["patch_size"]
        self.token_grid = self.img_size // self.patch_size
        self.num_prefix_tokens = getattr(self.encoder, "num_prefix_tokens", 1)
        pre_cfg = resolve_data_config(self.encoder.pretrained_cfg, model=self.encoder)
        self.register_buffer("mean", torch.tensor(pre_cfg["mean"]).view(1, 3, 1, 1),
                             persistent=False)
        self.register_buffer("std", torch.tensor(pre_cfg["std"]).view(1, 3, 1, 1),
                             persistent=False)

        n_genes = len(union_genes)
        if self.cell_feature_mode == "cross_attn":
            # Backbones wider than model_dim are first projected to model_dim.
            self.backbone_proj = (nn.Linear(self.backbone_dim, self.d)
                                  if self.backbone_dim != self.d else nn.Identity())
            self.kv_proj = nn.Linear(self.d, self.d)
            self.cross = CrossAttnRoPE(self.d, self.heads, self.drop)
            self.cell_norm = nn.LayerNorm(self.d)
            self.cls_attention = nn.MultiheadAttention(self.d, self.heads, dropout=self.drop,
                                                       batch_first=True)
            self.cls_layer_norm = nn.LayerNorm(self.d)
            head_dim = self.d
        else:
            head_dim = self.backbone_dim
        self.cell_gene = GeneHead(head_dim, 512, n_genes)
        self.cls_gene = GeneHead(head_dim, 512, n_genes)

        # Token-grid positions (x, y) in row-major token order.
        xs = torch.arange(self.token_grid)
        ys = torch.arange(self.token_grid)
        xv, yv = torch.meshgrid(xs, ys, indexing="xy")
        self.register_buffer("pos_tokens",
                             torch.stack([xv.reshape(-1), yv.reshape(-1)], dim=-1).long())

    # ------------------------------------------------------------------
    def encoder_parameters(self):
        return self.encoder.parameters()

    # ------------------------------------------------------------------
    def _cell_features(self, all_cell_xy, tokens_batch, cell_counts):
        """Grid-sample one query per cell, then (CELLO) refine it by cross-attention."""
        device = all_cell_xy.device
        B = tokens_batch.shape[0]
        dim = tokens_batch.shape[-1]
        cell_patch_idx = torch.repeat_interleave(
            torch.arange(B, device=device), torch.tensor(cell_counts, device=device))

        cell_tokens = tokens_batch[cell_patch_idx]                        # (N, M, C)
        cell_feats = cell_tokens.view(-1, self.token_grid, self.token_grid, dim).permute(0, 3, 1, 2)

        # CELLO Eq. 2: patch-local pixel coordinates -> [-1, 1].
        gx = (all_cell_xy[:, 0] / self.img_size) * 2 - 1
        gy = (all_cell_xy[:, 1] / self.img_size) * 2 - 1
        grid = torch.stack([gx, gy], dim=-1).view(-1, 1, 1, 2)
        q0 = F.grid_sample(cell_feats, grid, mode="bilinear",
                           align_corners=True).squeeze(-1).squeeze(-1)   # CELLO Eq. 3

        if self.cell_feature_mode == "grid":
            return q0

        bias = None
        if self.use_distance_bias:                                   # CELLO Eq. 4
            token_centers = (self.pos_tokens.float() + 0.5) * self.patch_size
            diff = token_centers.unsqueeze(0) - all_cell_xy.unsqueeze(1)
            bias = (-(diff ** 2).sum(-1) / (2.0 * self.patch_size ** 2)).clamp(min=-20.0)

        z = self.cross(q0, cell_tokens, self.pos_tokens, bias)
        return self.cell_norm(z)

    # ------------------------------------------------------------------
    @staticmethod
    def _unpack_cells(batch_data, device):
        cell_xy_pad = batch_data["cell_xy"].to(device)            # (B, max_n, 2)
        genes_target_pad = batch_data["genes_target"].to(device)  # (B, max_n, G)
        cell_mask = batch_data["cell_mask"].to(device)            # (B, max_n)
        cell_counts = [int(c) for c in batch_data["n_cells"]]
        return cell_xy_pad[cell_mask], genes_target_pad[cell_mask], cell_counts

    # ------------------------------------------------------------------
    def forward_batch(self, batch_data: Dict, union_index_map: Dict[str, int],
                      compute_preds: bool = False) -> Dict:
        device = self.pos_tokens.device
        imgs = batch_data["img"].to(device)
        B = imgs.shape[0]

        # 1. backbone (one forward pass per tile)
        x = (imgs - self.mean) / (self.std + 1e-6)
        feats = self.encoder.forward_features(x)                  # (B, prefix + M, C)
        if self.cell_feature_mode == "cross_attn":
            feats = self.backbone_proj(feats)
            cls_batch = feats[:, 0, :]
            tokens_batch = self.kv_proj(feats[:, self.num_prefix_tokens:, :])
        else:
            cls_batch = feats[:, 0, :]
            tokens_batch = feats[:, self.num_prefix_tokens:, :]

        all_cell_xy, all_genes_target, cell_counts = self._unpack_cells(batch_data, device)
        if all_cell_xy.shape[0] == 0:
            z = torch.tensor(0., device=device)
            return {"l_gene": z, "l_pg": z, "union_logits": None, "patch_pcc_list": [],
                    "val_cell_pred": None, "val_cell_gt": None}

        # 2. cell features
        z_cells = self._cell_features(all_cell_xy, tokens_batch, cell_counts)

        # 3. residual fusion with the tile CLS token (CELLO only)
        if self.cell_feature_mode == "cross_attn":
            cell_patch_idx = torch.repeat_interleave(
                torch.arange(B, device=device), torch.tensor(cell_counts, device=device))
            cls_for_cells = cls_batch[cell_patch_idx].unsqueeze(1)
            attn_out, _ = self.cls_attention(z_cells.unsqueeze(1), cls_for_cells, cls_for_cells)
            z_cells = self.cls_layer_norm(z_cells + attn_out.squeeze(1))

        # 4. cell-level gene prediction + loss over genes measured in this batch
        union_logits = self.cell_gene(z_cells)
        main_S_genes = batch_data["S_genes"]
        total_l_gene = torch.tensor(0., device=device)
        total_l_pg = torch.tensor(0., device=device)
        pred_S_all = torch.zeros(0, 0, device=device)

        if main_S_genes:
            idxs = torch.tensor([union_index_map[g] for g in main_S_genes],
                                dtype=torch.long, device=device)
            pred_S_all = union_logits[:, idxs]
            gene_exists = (all_genes_target.sum(0) > 0)
            if gene_exists.sum() > 0:
                mask = gene_exists.unsqueeze(0).expand_as(pred_S_all)
                total_l_gene = F.huber_loss(
                    pred_S_all[mask], all_genes_target[mask], delta=1.0).mean()

            # 5. patch-level auxiliary loss (CELLO Eq. 5): CLS -> sum of cell expressions in the tile
            cls_S = self.cls_gene(cls_batch)[:, idxs]
            patch_targets = torch.zeros(B, len(main_S_genes), device=device)
            s = 0
            for b in range(B):
                if cell_counts[b] > 0:
                    patch_targets[b] = all_genes_target[s:s + cell_counts[b]].sum(0)
                s += cell_counts[b]
            pe = (patch_targets.sum(0) > 0)
            if pe.sum() > 0:
                total_l_pg = F.huber_loss(cls_S[:, pe], patch_targets[:, pe], delta=1.0).mean()

        # 6. per-tile PCC (flattened over measured genes), used for quick monitoring
        patch_pcc_list = []
        if compute_preds and main_S_genes and pred_S_all.numel() > 0:
            s = 0
            for b in range(B):
                c = cell_counts[b]
                if c > 0:
                    pv = pred_S_all[s:s + c]
                    tv = all_genes_target[s:s + c]
                    exists = (tv.sum(0) > 0)
                    if exists.sum() > 1:
                        pf = pv[:, exists].reshape(-1)
                        tf = tv[:, exists].reshape(-1)
                        pc = pf - pf.mean()
                        tc = tf - tf.mean()
                        pcc = (pc * tc).sum() / (torch.sqrt((pc ** 2).sum() * (tc ** 2).sum()) + 1e-8)
                        patch_pcc_list.append(pcc.item())
                s += c

        return {
            "l_gene": total_l_gene,
            "l_pg": total_l_pg,
            "union_logits": union_logits if compute_preds else None,
            "patch_pcc_list": patch_pcc_list,
            "val_cell_pred": pred_S_all.detach() if compute_preds else None,
            "val_cell_gt": all_genes_target.detach() if compute_preds else None,
        }

    def forward(self, batch_data: Dict, union_index_map: Dict[str, int],
                compute_preds: bool = False) -> Dict:
        return self.forward_batch(batch_data, union_index_map, compute_preds)

    # ------------------------------------------------------------------
    @torch.no_grad()
    def predict(self, imgs: torch.Tensor, cell_xy: List[torch.Tensor]) -> List[torch.Tensor]:
        """Predict union-vocabulary expression for cells in a batch of tiles.

        imgs: (B, 3, H, W) float in [0, 1]. cell_xy: list of (n_i, 2) patch-local
        pixel coordinates. Returns a list of (n_i, G) tensors.
        """
        device = self.pos_tokens.device
        n_cells = [int(len(c)) for c in cell_xy]
        max_n = max(max(n_cells), 1)
        B = imgs.shape[0]
        xy_pad = torch.zeros(B, max_n, 2)
        mask = torch.zeros(B, max_n, dtype=torch.bool)
        for i, c in enumerate(cell_xy):
            xy_pad[i, :len(c)] = torch.as_tensor(c, dtype=torch.float32)
            mask[i, :len(c)] = True
        batch = {"img": imgs, "cell_xy": xy_pad, "cell_mask": mask, "n_cells": n_cells,
                 "genes_target": torch.zeros(B, max_n, 0), "S_genes": []}
        out = self.forward_batch(batch, {}, compute_preds=True)
        logits = out["union_logits"]
        if logits is None:
            return [torch.zeros(0, self.cell_gene.fc[-1].out_features) for _ in n_cells]
        return list(torch.split(logits.float().cpu(), n_cells))


# --------------------------------------------------------------------------
# Checkpoints
# --------------------------------------------------------------------------
def load_checkpoint(path: str, device="cpu", config_overrides: Optional[dict] = None,
                    pretrained_encoder: bool = False):
    """Build a CELLO model from a checkpoint written by ``scripts/train.py``.

    The backbone weights come from the checkpoint, so the PFM does not need to
    be downloaded again (``pretrained_encoder=False``); timm still needs the
    backbone definition from the Hugging Face hub.
    """
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    state = ckpt["model_state_dict"]
    ckpt.pop("model_state_dict", None)
    ckpt.pop("optimizer_state_dict", None)
    config = dict(ckpt.get("config", {}))
    if config_overrides:
        config.update(config_overrides)
    union_genes = list(ckpt["union_genes"])
    model = CELLO(union_genes, config, pretrained_encoder=pretrained_encoder)
    model.load_state_dict(state, strict=True)
    return model.to(device).eval(), config, union_genes
