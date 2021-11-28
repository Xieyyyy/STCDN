import torch
import torch.nn as nn
import torch.nn.functional as F


class GATLayer(nn.Module):
    def __init__(self, in_dim, out_dim, num_heads, feat_drop=0., attn_drop=0., neg_slope=0.1, activation=None):
        super(GATLayer, self).__init__()
        self.fc = nn.Linear(in_dim, out_dim * num_heads)
        self.num_heads = num_heads
        self.feat_drop = nn.Dropout(feat_drop)
        self.Q1 = nn.Parameter(
            torch.FloatTensor(
                size=(self.num_heads, out_dim, out_dim)))
        self.Q2 = nn.Parameter(
            torch.FloatTensor(size=(self.num_heads, out_dim, out_dim)))
        self.K = nn.Parameter(
            torch.FloatTensor(size=(self.num_heads, out_dim, out_dim)))
        self.V = nn.Parameter(
            torch.FloatTensor(size=(self.num_heads, out_dim, out_dim)))
        self.leaky_relu = nn.LeakyReLU(neg_slope)
        self.activation = nn.ReLU()

        self._reset_parameters()

    def _reset_parameters(self):
        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_uniform_(self.fc.weight, gain=gain)
        nn.init.xavier_uniform_(self.Q1, gain=gain)
        nn.init.xavier_uniform_(self.Q2, gain=gain)
        nn.init.xavier_uniform_(self.K, gain=gain)
        nn.init.xavier_uniform_(self.V, gain=gain)

    def forward(self, vt, x, adj):
        B, N, D = x.shape
        x = self.feat_drop(x)
        x = self.fc(x).reshape(B, N, self.num_heads, -1).transpose(1, 2)  # [32,8,170,16]
        Qt1 = x @ self.Q1  # [32,170,8,16]
        Qt2 = x @ self.Q2  # [32,170,8,16]
        Kt = x @ self.K  # [32,170,8,16]
        Vt = x @ self.V  # [32,170,8,16]
        Qt = self.leaky_relu(Qt1 + Qt2)
        QKt = self.leaky_relu(Qt @ Kt.transpose(-1, -2))
        AQKt = adj * QKt
        AQKt = torch.where(AQKt == 0, -9e15 * torch.ones_like(AQKt), AQKt)  # [32,8,170,170]
        AQKt = F.softmax(AQKt, dim=-1)  # [32,8,170,170]
        rst = AQKt @ Vt  # [32,8,170,16]
        rst = rst.transpose(1, 2)

        return self.activation(rst)


class GATEncoder(nn.Module):
    def __init__(self, args, in_dim, out_dim, num_layers, dropout, num_heads):
        super(GATEncoder, self).__init__()
        self.args = args
        self.nfe = 0
        self.initial_value = None
        self.num_layers = num_layers
        self.out_mlp = nn.Linear(num_layers * out_dim, out_dim)
        self.gat_layers = nn.ModuleList(
            GATLayer(in_dim=in_dim, out_dim=int(out_dim / num_heads), num_heads=num_heads,
                     feat_drop=dropout, attn_drop=dropout,
                     activation=F.relu) for _ in range(num_layers))

        self.dropout = dropout
        self.num_heads = num_heads

    def forward(self, vt=None, features=None, **kwargs):
        self.nfe += 1
        x = features
        out = []
        for idx, layer in enumerate(self.gat_layers):
            x = layer(vt, x, kwargs['adj_mx'])
            x = torch.cat([x[:, :, i, :] for i in range(self.num_heads)], dim=2)
            out.append(x)
        h = torch.cat(out, dim=-1)
        h = self.out_mlp(h)
        features = F.dropout(h, self.dropout, training=self.training)
        return features

    def reset(self):
        self.nfe = 0


class GATDecoder(nn.Module):
    def __init__(self, args, in_dim, out_dim, num_layers, dropout, num_heads):
        super(GATDecoder, self).__init__()
        self.args = args
        self.nfe = 0
        self.initial_value = None
        self.num_layers = num_layers
        self.out_mlp = nn.Linear(num_layers * out_dim, out_dim)
        self.gat_layers = nn.ModuleList(
            GATLayer(in_dim=in_dim, out_dim=int(out_dim / num_heads), num_heads=num_heads,
                     feat_drop=dropout, attn_drop=dropout,
                     activation=F.relu) for _ in range(num_layers))

        self.dropout = dropout
        self.num_heads = num_heads

    def forward(self, vt=None, features=None, **kwargs):
        self.nfe += 1
        if len(kwargs['solutions']) < self.args.back_look:
            back_look_feat = torch.cat([solution[0] for solution in kwargs['solutions']], dim=-1)
            pad_len = self.args.hidden_dim * (self.args.back_look - len(kwargs['solutions']))
            features = F.pad(back_look_feat, (pad_len, 0, 0, 0, 0, 0))
        else:
            features = torch.cat([solution[0] for solution in kwargs['solutions'][-self.args.back_look:]], dim=-1)
        x = features
        out = []
        for idx, layer in enumerate(self.gat_layers):
            x = layer(vt, x, kwargs['adj_mx'])
            x = torch.cat([x[:, :, i, :] for i in range(self.num_heads)], dim=2)
            out.append(x)
        h = torch.cat(out, dim=-1)
        h = self.out_mlp(h)
        features = F.dropout(h, self.dropout, training=self.training)
        return features

    def reset(self):
        self.nfe = 0
