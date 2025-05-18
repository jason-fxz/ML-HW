import torch
import torch.nn as nn
from torch.nn import functional as F
from einops import rearrange, repeat
from math import sqrt

from torch_geometric.nn import GATConv

class FullAttention(nn.Module):
    def __init__(self, scale=None, attention_dropout=0.1):
        super().__init__()
        self.scale = scale
        self.dropout = nn.Dropout(attention_dropout)
        
    def forward(self, queries, keys, values):
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        scale = self.scale or 1./sqrt(E)

        scores = torch.einsum("blhe,bshe->bhls", queries, keys)
        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        V = torch.einsum("bhls,bshd->blhd", A, values)
        
        return V.contiguous()

class AttentionLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_keys=None, d_values=None, dropout = 0.1):
        super().__init__()
        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)

        self.inner_attention = FullAttention(scale=None, attention_dropout=dropout)
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads

    def forward(self, queries, keys, values):
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads

        queries = self.query_projection(queries).view(B, L, H, -1)
        keys = self.key_projection(keys).view(B, S, H, -1)
        values = self.value_projection(values).view(B, S, H, -1)

        out = self.inner_attention(queries, keys, values)
        out = out.view(B, L, -1)

        return self.out_projection(out)

class GATAttentionLayer(nn.Module):
    def __init__(self, in_dim, out_dim=None, heads=4, dropout=0.1):
        super().__init__()
        out_dim = out_dim or in_dim
        self.gat = GATConv(in_dim, out_dim // heads, heads=heads, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(out_dim)
        self.ff = nn.Sequential(
            nn.Linear(out_dim, 4 * out_dim),
            nn.GELU(),
            nn.Linear(4 * out_dim, out_dim),
        )

    def forward(self, x, edge_index):
        '''
        x: [B, N, D], edge_index: [2, E]
        '''
        B, N, D = x.shape
        x_out = []
        for b in range(B):
            h = self.gat(x[b], edge_index)  # [N, D]
            h = self.dropout(h)
            h = self.norm(x[b] + h)
            h = self.norm(h + self.dropout(self.ff(h)))
            x_out.append(h)
        return torch.stack(x_out, dim=0)  # [B, N, D]

class TwoStageAttentionLayer(nn.Module):
    '''
    input/output shape: [batch_size, D (vars), L (seg), d_model]
    '''
    def __init__(self, seg_num, factor, d_model, n_heads, d_ff=None, dropout=0.1, use_gat=False):
        super().__init__()
        d_ff = d_ff or 4 * d_model
        self.use_gat = use_gat
        self.seg_num = seg_num

        self.time_attention = AttentionLayer(d_model, n_heads, dropout=dropout)
        self.dropout = nn.Dropout(dropout)

        # GAT 组件（只用于 cross-dim）
        if use_gat:
            self.dim_gat = GATAttentionLayer(d_model, d_model, heads=n_heads, dropout=dropout)
            self.edge_index = None
        else:
            self.dim_sender = AttentionLayer(d_model, n_heads, dropout=dropout)
            self.dim_receiver = AttentionLayer(d_model, n_heads, dropout=dropout)
            self.router = nn.Parameter(torch.randn(seg_num, factor, d_model))

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.norm4 = nn.LayerNorm(d_model)

        self.MLP1 = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model)
        )
        self.MLP2 = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model)
        )

    def build_fully_connected_graph(self, num_nodes):
        edge_index = []
        for i in range(num_nodes):
            for j in range(num_nodes):
                if i != j:
                    edge_index.append([i, j])
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        return edge_index  # shape [2, E]

    def forward(self, x):
        # x: [B, D, L, d_model]
        B, D, L, C = x.shape

        # Time attention (Cross-Time Stage)
        time_in = rearrange(x, 'b d l c -> (b d) l c')
        time_enc = self.time_attention(time_in, time_in, time_in)
        time_enc = self.norm1(time_in + self.dropout(time_enc))
        time_enc = self.norm2(time_enc + self.dropout(self.MLP1(time_enc)))

        # Reshape to [B * L, D, d_model] for Cross-Dimension
        if self.use_gat:
            D = x.shape[1]
            dim_in = rearrange(time_enc, '(b d) l c -> (b l) d c', b=B)
            if self.edge_index is None:
                self.edge_index = self.build_fully_connected_graph(D).to(x.device)
            dim_enc = self.dim_gat(dim_in, self.edge_index.to(x.device))
        else:
            dim_in = rearrange(time_enc, '(b d) l c -> (b l) d c', b=B)
            batch_router = repeat(self.router, 'l f c -> (b l) f c', b=B)
            dim_buffer = self.dim_sender(batch_router, dim_in, dim_in)
            dim_receive = self.dim_receiver(dim_in, dim_buffer, dim_buffer)
            dim_enc = self.norm3(dim_in + self.dropout(dim_receive))
            dim_enc = self.norm4(dim_enc + self.dropout(self.MLP2(dim_enc)))

        # Reshape back to [B, D, L, d_model]
        final_out = rearrange(dim_enc, '(b l) d c -> b d l c', b=B)
        return final_out
