import dgl.function as fn
import dgl.ops as ops
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.utils import expand_as_pair


class GATLayer(nn.Module):
    def __init__(self, in_dim, out_dim, num_heads, graph, feat_drop=0., attn_drop=0., neg_slope=0.1, activation=None):
        super(GATLayer, self).__init__()
        self.num_heads = num_heads
        self.in_src_dim, self.in_dst_dim = expand_as_pair(in_dim)
        self.out_dim = out_dim
        self.feat_drop = nn.Dropout(feat_drop)
        self.attn_drop = nn.Dropout(attn_drop)
        self.leaky_relu = nn.LeakyReLU(neg_slope)
        self.activation = activation
        self.graph = graph
        self.fc = nn.Linear(self.in_src_dim, self.in_src_dim, bias=False)
        self.attn_left = nn.Parameter(torch.FloatTensor(size=(1, self.num_heads, int(self.in_src_dim / num_heads))))
        self.attn_right = nn.Parameter(torch.FloatTensor(size=(1, self.num_heads, int(self.in_src_dim / num_heads))))
        self.out_fc = nn.Linear(int(self.in_src_dim / num_heads), self.out_dim)
        self._reset_parameters()

    def _reset_parameters(self):
        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_uniform_(self.fc.weight, gain=gain)
        nn.init.xavier_uniform_(self.attn_right, gain=gain)
        nn.init.xavier_uniform_(self.attn_left, gain=gain)

    def forward(self, vt, x):
        with self.graph.local_scope():
            B, N, D = x.shape
            h_src = self.feat_drop(x)
            feat_src = feat_dst = self.fc(h_src).view(B, N, self.num_heads, -1).transpose(0, 1)  # [170,32,8,16]

            el = (feat_src * self.attn_left).sum(dim=-1).unsqueeze(-1)
            er = (feat_dst * self.attn_right).sum(dim=-1).unsqueeze(-1)
            self.graph.srcdata.update({'ft': feat_src, 'el': el})
            self.graph.dstdata.update({'er': er})
            self.graph.apply_edges(fn.u_add_v(lhs_field='el', rhs_field='er', out='e'))
            e = self.leaky_relu(self.graph.edata.pop('e'))  # [edge_num,batch_size,num_head,1]
            w = self.graph.edata['w']
            w = w.reshape(w.shape[0], 1, 1, 1).repeat(1, B, self.num_heads, 1)
            e = w * e
            self.graph.edata['a'] = self.attn_drop(ops.edge_softmax(self.graph, e))
            # self.graph.edata['a'] = self.attn_drop(e)

            self.graph.update_all(fn.u_mul_e(lhs_field='ft', rhs_field='a', out='m'), fn.sum(msg='m', out='ft'))
            rst = self.graph.dstdata['ft']
            rst = self.out_fc(rst)

            if self.activation:
                rst = self.activation(rst)

        return rst.transpose(0, 1)


class GATEncoder(nn.Module):
    def __init__(self, args, in_dim, out_dim, num_layers, dropout, num_heads, graph):
        super(GATEncoder, self).__init__()
        self.args = args
        self.nfe = 0
        self.initial_value = None
        self.num_layers = num_layers
        self.out_mlp = nn.Linear(num_layers * out_dim, out_dim)
        self.gat_layers = nn.ModuleList(
            GATLayer(in_dim=in_dim, out_dim=int(out_dim / num_heads), num_heads=num_heads,
                     graph=graph,
                     feat_drop=dropout, attn_drop=dropout,
                     activation=F.relu) for _ in range(num_layers))

        self.dropout = dropout
        self.num_heads = num_heads

    def forward(self, vt=None, features=None, solutions=None):
        self.nfe += 1
        x = features
        out = []
        for idx, layer in enumerate(self.gat_layers):
            x = layer(vt, x)
            x = torch.cat([x[:, :, i, :] for i in range(self.num_heads)], dim=2)
            out.append(x)
        h = torch.cat(out, dim=-1)
        h = self.out_mlp(h)
        features = F.dropout(h, self.dropout, training=self.training)
        return features

    def reset(self):
        self.nfe = 0


class GATDecoder(nn.Module):
    def __init__(self, args, in_dim, out_dim, num_layers, dropout, num_heads, graph):
        super(GATDecoder, self).__init__()
        self.args = args
        self.nfe = 0
        self.initial_value = None
        self.num_layers = num_layers
        self.out_mlp = nn.Linear(num_layers * out_dim, out_dim)
        self.gat_layers = nn.ModuleList(
            GATLayer(in_dim=in_dim, out_dim=int(out_dim / num_heads), num_heads=num_heads,
                     graph=graph,
                     feat_drop=dropout, attn_drop=dropout,
                     activation=F.relu) for _ in range(num_layers))

        self.dropout = dropout
        self.num_heads = num_heads

    def forward(self, vt=None, features=None, solutions=None):
        self.nfe += 1
        if len(solutions) < self.args.back_look:
            back_look_feat = torch.cat([solution[0] for solution in solutions], dim=-1)
            pad_len = self.args.hidden_dim * (self.args.back_look - len(solutions))
            features = F.pad(back_look_feat, (pad_len, 0, 0, 0, 0, 0))
        else:
            features = torch.cat([solution[0] for solution in solutions[-self.args.back_look:]], dim=-1)
        x = features
        out = []
        for idx, layer in enumerate(self.gat_layers):
            x = layer(vt, x)
            x = torch.cat([x[:, :, i, :] for i in range(self.num_heads)], dim=2)
            out.append(x)
        h = torch.cat(out, dim=-1)
        h = self.out_mlp(h)
        features = F.dropout(h, self.dropout, training=self.training)
        return features

    def reset(self):
        self.nfe = 0
