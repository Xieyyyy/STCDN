import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import sparse as sp


class GraphLearner(nn.Module):
    def __init__(self, args):
        super(GraphLearner, self).__init__()
        self.args = args
        self.M1 = nn.Parameter(torch.FloatTensor(size=(self.args.num_node, 64)), requires_grad=True)
        self.M2 = nn.Parameter(torch.FloatTensor(size=(self.args.num_node, 64)), requires_grad=True)
        self.k = int(self.args.retain_ratio * self.args.num_node * self.args.num_node)
        self._reset_parameters()

    def _reset_parameters(self):
        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_uniform_(self.M1, gain=gain)
        nn.init.xavier_uniform_(self.M2, gain=gain)

    def forward(self, x):
        adj_mx = F.relu(torch.matmul(self.M1, self.M2.transpose(0, 1)))
        adj_mx = F.sigmoid(adj_mx)
        edges_w = adj_mx.view(self.args.num_node * self.args.num_node)
        k = torch.topk(edges_w, self.k)[0].min()
        adj_mx = torch.where(adj_mx > k, adj_mx, torch.zeros_like(adj_mx))
        adj_mx = sp.csr_matrix(adj_mx.cpu().detach().numpy())
        graph = dgl.from_scipy(adj_mx, eweight_name='w')
        return graph.to(self.M1.device)
