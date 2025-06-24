import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.utils import to_undirected
from torch_geometric.nn import knn_graph
from einops import rearrange
import math
from timm.layers import DropPath, trunc_normal_
from operator import mul
from functools import reduce

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Sinkhorn Matching Loss
class SinkhornMatchingLoss(nn.Module):
    def __init__(self, num_iter=5, tau=0.05, eps=1e-8):
        super().__init__()
        self.num_iter = num_iter
        self.tau = tau
        self.eps = eps

    def forward(self, source_nodes, target_nodes):
        B, T, N, D = source_nodes.shape
        src = source_nodes.view(B * T, N, D)
        tgt = target_nodes.view(B * T, N, D)
        cost_matrix = torch.cdist(src, tgt, p=2)
        K = torch.exp(-cost_matrix / self.tau)
        mu = torch.full((B * T, N), 1.0 / N, device=device)
        nu = torch.full((B * T, N), 1.0 / N, device=device)
        u = torch.ones_like(mu)
        v = torch.ones_like(nu)
        for _ in range(self.num_iter):
            u = mu / (torch.bmm(K, v.unsqueeze(-1)).squeeze(-1) + self.eps)
            v = nu / (torch.bmm(K.transpose(1, 2), u.unsqueeze(-1)).squeeze(-1) + self.eps)
        pi = u.unsqueeze(-1) * K * v.unsqueeze(-2)
        frame_loss = torch.sum(pi * cost_matrix, dim=(1, 2))
        total_loss = frame_loss.mean()
        return total_loss

# Patch Embedding
class PatchEmbed3D(nn.Module):
    def __init__(self, patch_size=(1,4,4), in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        self.patch_size = patch_size

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.proj = nn.Conv3d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        _, _, D, H, W = x.size()
        if W % self.patch_size[2] != 0:
            x = F.pad(x, (0, self.patch_size[2] - W % self.patch_size[2]))
        if H % self.patch_size[1] != 0:
            x = F.pad(x, (0, 0, 0, self.patch_size[1] - H % self.patch_size[1]))
        if D % self.patch_size[0] != 0:
            x = F.pad(x, (0, 0, 0, 0, 0, self.patch_size[0] - D % self.patch_size[0]))

        x = self.proj(x)  # B C D Wh Ww
        if self.norm is not None:
            D, Wh, Ww = x.size(2), x.size(3), x.size(4)
            x = x.flatten(2).transpose(1, 2)
            x = self.norm(x)
            x = x.transpose(1, 2).view(-1, self.embed_dim, D, Wh, Ww)

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
        embed_dim = self.emb_t.embedding_dim
        pos_emb_t = self.emb_t(pos_t).permute(1, 0).view(1, embed_dim, t, 1, 1) * self.scale
        pos_emb_h = self.emb_h(pos_h).permute(1, 0).view(1, embed_dim, 1, h, 1) * self.scale
        pos_emb_w = self.emb_w(pos_w).permute(1, 0).view(1, embed_dim, 1, 1, w) * self.scale
        x = x + pos_emb_t + pos_emb_h + pos_emb_w
        if self.post_norm is not None:
            x = rearrange(x, 'b c t h w -> b t h w c')
            x = self.post_norm(x)
            x = rearrange(x, 'b t h w c -> b c t h w')
        return x

# Swin Transformer Components
def window_partition(x, window_size):
    B, D, H, W, C = x.shape
    x = x.view(B, D // window_size[0], window_size[0], H // window_size[1], window_size[1], W // window_size[2], window_size[2], C)
    windows = x.permute(0, 1, 3, 5, 2, 4, 6, 7).contiguous().view(-1, reduce(mul, window_size), C)
    return windows

def window_reverse(windows, window_size, B, D, H, W):
    x = windows.view(B, D // window_size[0], H // window_size[1], W // window_size[2], window_size[0], window_size[1], window_size[2], -1)
    x = x.permute(0, 1, 4, 2, 5, 3, 6, 7).contiguous().view(B, D, H, W, -1)
    return x

def get_window_size(x_size, window_size, shift_size=None):
    use_window_size = list(window_size)
    if shift_size is not None:
        use_shift_size = list(shift_size)
    for i in range(len(x_size)):
        if x_size[i] <= window_size[i]:
            use_window_size[i] = x_size[i]
            if shift_size is not None:
                use_shift_size[i] = 0

    if shift_size is None:
        return tuple(use_window_size)
    else:
        return tuple(use_window_size), tuple(use_shift_size)

class WindowAttention3D(nn.Module):
    def __init__(self, dim, window_size, num_heads, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, batch_size=8, frame_len=8):
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        scale = 1.0 / math.sqrt(C // self.num_heads)
        scores = torch.einsum("b h n c, b h m c -> b h n m", q, k) * scale
        attn = F.softmax(scores, dim=-1)
        if self.training:
            attn = F.dropout(attn, p=self.attn_drop.p)
        self.last_attn = attn
        attn_output = torch.einsum("b h n m, b h m c -> b h n c", attn, v)
        attn_output = attn_output.transpose(1, 2).reshape(B_, N, C)
        x = self.proj(attn_output)
        x = self.proj_drop(x)
        return x

class MLP(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer='swish', drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.SiLU() if act_layer == 'swish' else nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class SwinTransformerBlock3D(nn.Module):
    def __init__(self, dim, num_heads, window_size=(1,4,4), shift_size=(0,0,0), mlp_ratio=4., qkv_bias=True,
                 drop=0., attn_drop=0., drop_path=0., act_layer='swish', norm_layer=nn.LayerNorm,
                 use_checkpoint=False):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        self.use_checkpoint = use_checkpoint
        self.norm1 = norm_layer(dim)

        assert 0 <= self.shift_size[0] < self.window_size[0], "shift_size must in 0-window_size"
        assert 0 <= self.shift_size[1] < self.window_size[1], "shift_size must in 0-window_size"
        assert 0 <= self.shift_size[2] < self.window_size[2], "shift_size must in 0-window_size"
                     
        self.attn = WindowAttention3D(
            dim, window_size=self.window_size, num_heads=num_heads,
            qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
                     
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward_part1(self, x):
        B, D, H, W, C = x.shape
        window_size, shift_size = get_window_size((D, H, W), self.window_size, self.shift_size)
        x = self.norm1(x)
        pad_l = pad_t = pad_d0 = 0
        pad_d1 = (window_size[0] - D % window_size[0]) % window_size[0]
        pad_b = (window_size[1] - H % window_size[1]) % window_size[1]
        pad_r = (window_size[2] - W % window_size[2]) % window_size[2]
        x = F.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b, pad_d0, pad_d1))
        _, Dp, Hp, Wp, _ = x.shape
        if any(i > 0 for i in shift_size):
            shifted_x = torch.roll(x, shifts=(-shift_size[0], -shift_size[1], -shift_size[2]), dims=(1, 2, 3))
        else:
            shifted_x = x
        x_windows = window_partition(shifted_x, window_size)
        attn_windows = self.attn(x_windows, batch_size=B, frame_len=D)
        attn_windows = attn_windows.view(-1, *(window_size+(C,)))
        shifted_x = window_reverse(attn_windows, window_size, B, Dp, Hp, Wp)
        if any(i > 0 for i in shift_size):
            x = torch.roll(shifted_x, shifts=(shift_size[0], shift_size[1], shift_size[2]), dims=(1, 2, 3))
        else:
            x = shifted_x
        if pad_d1 > 0 or pad_r > 0 or pad_b > 0:
            x = x[:, :D, :H, :W, :].contiguous()
        return x

    def forward_part2(self, x):
        return self.drop_path(self.mlp(self.norm2(x)))

    def forward(self, x):
        shortcut = x
        x = self.forward_part1(x)
        x = shortcut + self.drop_path(x)
        x = x + self.forward_part2(x)
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
        patch_size=(1, 16, 16),
        input_resolution=128,
        embed_dim=192,
        act_layer='swish',
        depth=12,
        num_heads=8,
        window_size=(1,4,4),
        mlp_ratio=4.,
        gcn_dim=128,
        rgcc_hidden_dim=128,
        k_neighbors=5,
        num_frames=180,
        norm_layer=nn.LayerNorm,
        drop_rate=0.0,
        attn_drop_rate=0.,
        drop_path_rate=0.2,
        post_pos_norm=True,
        use_checkpoint=False,
        patch_norm=False,
        qkv_bias=True,
        **kwargs,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.num_frames = num_frames
        self.k_neighbors = k_neighbors

        self.patch_embed = PatchEmbed3D(
            patch_size=patch_size, in_chans=3, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)
        
        self.pos_embed = AbsolutePositionalEmbedding(
            embed_dim,
            max_seq_len=(1800, math.ceil(input_resolution/patch_size[1]), math.ceil(input_resolution/patch_size[2])),
            post_norm=norm_layer if post_pos_norm else None,
        )
        self.pos_drop = nn.Dropout(p=drop_rate)
        drop_path_rate = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]

        self.shift_size = tuple(i // 2 for i in window_size)
        
        self.swin_transformer = nn.ModuleList([
            SwinTransformerBlock3D(
                dim=dim,
                num_heads=num_heads,
                window_size=window_size,
                shift_size=(0,0,0) if (i % 2 == 0) else self.shift_size,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=drop_path[i] if isinstance(drop_path_rate, list) else drop_path_rate,
                act_layer=act_layer,
                norm_layer=norm_layer,
                use_checkpoint=use_checkpoint,
            )
            for i in range(depth)])
        
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
        
        # Qua Swin Transformer
        src = rearrange(src, 'b c t h w -> b t h w c')
        src = self.swin_transformer(src)
        src = rearrange(src, 'b t h w c -> b c t h w')
        
        _, feat_dim, T_p, H_p, W_p = src.shape
        N = H_p * W_p
        src_nodes, src_edges = self.build_local_graph(src)
        src_nodes = self.gcn(src_nodes, src_edges)

        matching_loss = None
        if target_x is not None:
            tgt = self.preprocess(target_x)
            tgt = self.patch_embed(tgt)
            tgt = self.pos_drop(self.pos_embed(tgt))
            tgt = rearrange(tgt, 'b c t h w -> b t h w c')
            tgt = self.swin_transformer(tgt)
            tgt = rearrange(tgt, 'b t h w c -> b c t h w')
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
