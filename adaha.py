import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.utils import to_undirected
from torch_geometric.nn import knn_graph
from einops import rearrange
import math
from timm.layers import DropPath, trunc_normal_

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Sinkhorn Matching Loss
class SinkhornMatchingLoss(nn.Module):
    def __init__(self, num_iter=10, tau=0.05, eps=1e-8):
        super().__init__()
        self.num_iter = num_iter
        self.tau = tau
        self.eps = eps

    def forward(self, source_nodes, target_nodes):
        B, T, N, D = source_nodes.shape
        total_loss = 0.0

        # Xử lý từng batch và frame để tránh out-of-memory
        for b in range(B):
            for t in range(T):
                src = source_nodes[b, t]  # [N, D]
                tgt = target_nodes[b, t]  # [N, D]
                cost_matrix = torch.cdist(src, tgt, p=2)  # [N, N]
                K = torch.exp(-cost_matrix / self.tau)

                mu = torch.full((N,), 1.0 / N, device=device)
                nu = torch.full((N,), 1.0 / N, device=device)
                u = torch.ones_like(mu)
                v = torch.ones_like(nu)

                for _ in range(self.num_iter):
                    u = mu / (K @ v + self.eps)
                    v = nu / (K.t() @ u + self.eps)

                pi = torch.diag(u) @ K @ torch.diag(v)
                frame_loss = torch.sum(pi * cost_matrix)
                total_loss += frame_loss

        return total_loss / (B * T)

# Patch Embedding
class PatchEmbed3D(nn.Module):
    def __init__(self, patch_size=(1, 32, 32), in_chans=3, embed_dim=192, norm_layer=None):
        super().__init__()
        self.patch_size = patch_size
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        self.proj = nn.Conv3d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer else None

    def forward(self, x):
        _, _, T, H, W = x.size()
        for i, (dim, patch_dim) in enumerate(zip([T, H, W], self.patch_size)):
            if dim % patch_dim != 0:
                pad = [0] * (2 * 5)
                pad[2 * (4 - i) + 1] = patch_dim - dim % patch_dim
                x = F.pad(x, pad)
        x = self.proj(x)
        if self.norm:
            T_p, H_p, W_p = x.size(2), x.size(3), x.size(4)
            x = x.flatten(2).transpose(1, 2)
            x = self.norm(x)
            x = x.transpose(1, 2).view(-1, self.embed_dim, T_p, H_p, W_p)
        return x

# Positional Embedding
class AbsolutePositionalEmbedding(nn.Module):
    def __init__(self, dim, max_seq_len, post_norm=None):
        super().__init__()
        self.scale = dim ** -0.5
        self.max_t, self.max_h, self.max_w = max_seq_len
        self.emb_t = nn.Embedding(self.max_t, dim)
        self.emb_h = nn.Embedding(self.max_h, dim)
        self.emb_w = nn.Embedding(self.max_w, dim)
        self.post_norm = post_norm(dim) if callable(post_norm) else None

    def forward(self, x):
        t, h, w = x.shape[-3:]
        pos_t = torch.arange(t, device=x.device)
        pos_h = torch.arange(h, device=x.device)
        pos_w = torch.arange(w, device=x.device)
        embed_dim = self.emb_t.embedding_dim  # Lấy embed_dim từ embedding layer

        # Sửa shape của pos_emb_t, pos_emb_h, pos_emb_w
        pos_emb_t = self.emb_t(pos_t).permute(1, 0).view(1, embed_dim, t, 1, 1) * self.scale
        pos_emb_h = self.emb_h(pos_h).permute(1, 0).view(1, embed_dim, 1, h, 1) * self.scale
        pos_emb_w = self.emb_w(pos_w).permute(1, 0).view(1, embed_dim, 1, 1, w) * self.scale

        x = x + pos_emb_t + pos_emb_h + pos_emb_w

        if self.post_norm is not None:
            x = rearrange(x, 'b c t h w -> b t h w c')
            x = self.post_norm(x)
            x = rearrange(x, 'b t h w c -> b c t h w')

        return x

# Graph Convolutional Block
class GCNBlock(nn.Module):
    def __init__(self, in_dim, out_dim, num_layers=2):
        super().__init__()
        self.layers = nn.ModuleList([GCNConv(in_dim if i == 0 else out_dim, out_dim)
                                   for i in range(num_layers)])
        self.norm = nn.BatchNorm1d(out_dim)

    def forward(self, x, edge_index):
        for layer in self.layers:
            x = layer(x, edge_index)
            x = F.relu(x)
        return self.norm(x)

# Recurrent Graph Convolutional Network (RGCC)
class RGCC(nn.Module):
    def __init__(self, in_dim, hidden_dim, num_layers=2):
        super().__init__()
        self.layers = nn.ModuleList([GCNConv(in_dim if i == 0 else hidden_dim, hidden_dim)
                                   for i in range(num_layers)])
        self.gru = nn.GRU(hidden_dim, hidden_dim, num_layers=1, batch_first=True)

    def forward(self, x, edge_index):
        B, T, C = x.shape
        x_flat = x.view(B * T, C)
        for layer in self.layers:
            x_flat = layer(x_flat, edge_index)
            x_flat = F.relu(x_flat)
        x = x_flat.view(B, T, -1)
        x, _ = self.gru(x)
        return x

# Main Model Class
class Reperio(nn.Module):
    def __init__(
        self,
        patch_size=(1, 32, 32),
        input_resolution=128,
        embed_dim=192,
        gcn_dim=128,
        rgcc_hidden_dim=128,
        k_neighbors=5,
        num_frames=180,
        norm_layer=nn.LayerNorm,
        drop_rate=0.0
    ):
        super().__init__()
        self.patch_size = patch_size
        self.num_frames = num_frames
        self.k_neighbors = k_neighbors

        self.patch_embed = PatchEmbed3D(patch_size=patch_size, in_chans=3, embed_dim=embed_dim, norm_layer=norm_layer)
        self.pos_embed = AbsolutePositionalEmbedding(embed_dim, (num_frames, input_resolution // patch_size[1], input_resolution // patch_size[2]), norm_layer)
        self.pos_drop = nn.Dropout(p=drop_rate)
        self.gcn = GCNBlock(embed_dim, gcn_dim)
        self.rgcc = RGCC(gcn_dim, rgcc_hidden_dim)
        self.final_linear = nn.Linear(rgcc_hidden_dim, 1)
        self.sinkhorn_loss = SinkhornMatchingLoss()
        self.bn = nn.BatchNorm3d(3)
        self.apply(self._init_weights)

    def preprocess(self, x):
        x = x.permute(0, 2, 1, 3, 4)  # [B, T, C, H, W] -> [B, C, T, H, W]
        x = self.bn(x)
        return x

    def build_local_graph(self, x):
        B, C, T, H, W = x.shape
        N = H * W
        nodes = rearrange(x, 'b c t h w -> (b t h w) c')
        batch = torch.arange(B * T, device=x.device).repeat_interleave(N)
        edge_index = knn_graph(nodes, k=self.k_neighbors, batch=batch, loop=True)
        return nodes, edge_index

    def build_temporal_graph(self, B, T):
        edges = []
        for b in range(B):
            for t in range(T - 1):
                edges.append([b * T + t, b * T + t + 1])
                edges.append([b * T + t + 1, b * T + t])
        edge_index = torch.tensor(edges, dtype=torch.long, device=device).T
        return to_undirected(edge_index)

    def forward(self, source_x, target_x=None, labels=None, return_matching_loss=False):
        src = self.preprocess(source_x)
        B, C, T, H, W = src.shape
        src = self.patch_embed(src)
        src = self.pos_drop(self.pos_embed(src))
        _, feat_dim, T_p, H_p, W_p = src.shape
        N = H_p * W_p
        src_nodes, src_edges = self.build_local_graph(src)
        src_nodes = self.gcn(src_nodes, src_edges)

        matching_loss = None
        if target_x is not None:
            tgt = self.preprocess(target_x)
            tgt = self.patch_embed(tgt)
            tgt = self.pos_drop(self.pos_embed(tgt))
            tgt_nodes, tgt_edges = self.build_local_graph(tgt)
            tgt_nodes = self.gcn(tgt_nodes, tgt_edges)
            src_4d = src_nodes.view(B, T, N, -1)
            tgt_4d = tgt_nodes.view(B, T, N, -1)
            matching_loss = self.sinkhorn_loss(src_4d, tgt_4d)

        batch_index = torch.arange(B * T, device=src_nodes.device).repeat_interleave(N)
        pooled = global_mean_pool(src_nodes, batch_index)
        source_pooled = pooled.view(B, T, -1)
        temporal_edges = self.build_temporal_graph(B, T)
        temporal_feats = self.rgcc(source_pooled, temporal_edges)
        preds = self.final_linear(temporal_feats).squeeze(-1)
        
        return preds, matching_loss


    @torch.no_grad()  
    def predict(self, x):
        preds, _ = self.forward(x)
        return preds

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
