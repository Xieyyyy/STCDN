import torch
import torch.nn as nn
import torch.nn.functional as F


class GraphLearner(nn.Module):
    def __init__(self, args):
        super(GraphLearner, self).__init__()
        self.args = args
        self.M1 = nn.Parameter(torch.FloatTensor(size=(self.args.num_node, 64)), requires_grad=True)
        self.M2 = nn.Parameter(torch.FloatTensor(size=(self.args.num_node, 64)), requires_grad=True)
        self.k = int(self.args.retain_ratio * self.args.num_node * self.args.num_node)
        eye = torch.eye(self.args.num_node)
        self.register_buffer('eye', eye)
        self._reset_parameters()

    def _reset_parameters(self):
        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_uniform_(self.M1, gain=gain)
        nn.init.xavier_uniform_(self.M2, gain=gain)

    def forward(self, x):
        adj_mx = F.relu(torch.matmul(self.M1, self.M2.transpose(0, 1)))
        edges_w = adj_mx.view(self.args.num_node * self.args.num_node)
        k = torch.topk(edges_w, self.k)[0].min()
        adj_mx = torch.where(adj_mx > k, adj_mx, -9e15 * torch.ones_like(adj_mx))
        adj_mx = F.softmax(adj_mx, dim=-1)
        adj_mx = adj_mx * (1 - self.eye) + self.eye
        return adj_mx
