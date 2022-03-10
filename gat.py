import dgl.function as fn
import dgl.ops as ops
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.utils import expand_as_pair


class GATLayer(nn.Module):
    def __init__(self, in_dim, out_dim, num_heads, feat_drop=0., attn_drop=0., neg_slope=0.1, activation=None):
        super(GATLayer, self).__init__()
        self.num_heads = num_heads
        self.in_src_dim, self.in_dst_dim = expand_as_pair(in_dim)
        self.out_dim = out_dim
        self.feat_drop = nn.Dropout(feat_drop)
        self.attn_drop = nn.Dropout(attn_drop)
        self.leaky_relu = nn.LeakyReLU(neg_slope)
        self.activation = activation
        self.fc = nn.Linear(self.in_src_dim, self.in_src_dim)
        self.attn_left = nn.Parameter(
            torch.FloatTensor(size=(1, self.num_heads, int(self.in_src_dim / self.num_heads))))
        self.attn_right = nn.Parameter(
            torch.FloatTensor(size=(1, self.num_heads, int(self.in_src_dim / self.num_heads))))
        self.out_fc = nn.Linear(int(self.in_src_dim / self.num_heads), self.out_dim)
        self._reset_parameters()

    def _reset_parameters(self):
        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_uniform_(self.fc.weight, gain=gain)
        nn.init.xavier_uniform_(self.attn_right, gain=gain)
        nn.init.xavier_uniform_(self.attn_left, gain=gain)

    def forward(self, vt, x, graph):
        with graph.local_scope():
            B, N, D = x.shape
            h_src = self.feat_drop(x)
            feat_src = feat_dst = self.fc(h_src).view(B, N, self.num_heads, -1).transpose(0, 1)  # [170,32,8,16]

            el = (feat_src * self.attn_left).sum(dim=-1).unsqueeze(-1)  # attn_left:[1,8,16]
            er = (feat_dst * self.attn_right).sum(dim=-1).unsqueeze(-1)
            graph.srcdata.update({'ft': feat_src, 'el': el})
            graph.dstdata.update({'er': er})
            graph.apply_edges(fn.u_add_v(lhs_field='el', rhs_field='er', out='e'))
            e = self.leaky_relu(graph.edata.pop('e'))  # [edge_num,batch_size,num_head,1]
            w = graph.edata['w']
            w = w.reshape(w.shape[0], 1, 1, 1).repeat(1, B, self.num_heads, 1)
            e = w * e
            graph.edata['a'] = self.attn_drop(ops.edge_softmax(graph, e))
            # self.graph.edata['a'] = self.attn_drop(e)

            graph.update_all(fn.u_mul_e(lhs_field='ft', rhs_field='a', out='m'), fn.sum(msg='m', out='ft'))
            rst = graph.dstdata['ft']
            rst = self.out_fc(rst)

            if self.activation:
                rst = self.activation(rst)

        return rst.transpose(0, 1)


class GATEncoder(nn.Module):
    def __init__(self, args, in_dim, out_dim, num_layers, dropout, num_heads):
        super(GATEncoder, self).__init__()
        self.args = args
        self.nfe = 0
        self.initial_value = None
        self.num_layers = num_layers
        self.out_mlp = nn.Linear(num_layers * out_dim, out_dim)
        # self.actication = nn.PReLU()
        self.gat_layers = nn.ModuleList(
            GATLayer(in_dim=in_dim, out_dim=int(out_dim / num_heads), num_heads=num_heads,
                     feat_drop=dropout, attn_drop=dropout,
                     activation=nn.PReLU()) for _ in range(num_layers))

        self.dropout = dropout
        self.num_heads = num_heads
        self.graph = None

    def forward(self, vt=None, features=None):
        self.nfe += 1
        x = features
        out = []
        for idx, layer in enumerate(self.gat_layers):
            x = layer(vt, x, self.graph)
            x = torch.cat([x[:, :, i, :] for i in range(self.num_heads)], dim=2)
            out.append(x)
        h = torch.cat(out, dim=-1)
        h = self.out_mlp(h)
        features = F.dropout(h, self.dropout, training=self.training)
        return features

    def reset(self):
        self.nfe = 0

    def set_graph(self, graph):
        self.graph = graph


class GATDecoder(nn.Module):
    def __init__(self, args, in_dim, out_dim, num_layers, dropout, num_heads):
        super(GATDecoder, self).__init__()
        self.args = args
        self.nfe = 0
        self.initial_value = None
        self.num_layers = num_layers
        self.out_mlp = nn.Linear(num_layers * out_dim, out_dim)
        # self.actication = nn.PReLU()
        self.gat_layers = nn.ModuleList(
            GATLayer(in_dim=in_dim, out_dim=int(out_dim / num_heads), num_heads=num_heads,
                     feat_drop=dropout, attn_drop=dropout,
                     activation=nn.PReLU()) for _ in range(num_layers))

        self.dropout = dropout
        self.num_heads = num_heads
        self.graph = None

    def forward(self, vt=None, features=None):
        self.nfe += 1
        x = features
        out = []
        for idx, layer in enumerate(self.gat_layers):
            x = layer(vt, x, self.graph)
            x = torch.cat([x[:, :, i, :] for i in range(self.num_heads)], dim=2)
            out.append(x)
        h = torch.cat(out, dim=-1)
        h = self.out_mlp(h)
        features = F.dropout(h, self.dropout, training=self.training)
        return features

    def reset(self):
        self.nfe = 0

    def set_graph(self, graph):
        self.graph = graph
