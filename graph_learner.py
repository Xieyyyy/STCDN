import copy

import torch
import torch.nn as nn
import torch.nn.functional as F


class GraphLearner(nn.Module):
    def __init__(self, args, complete_graph_structure):
        super(GraphLearner, self).__init__()
        self.args = args
        self.complete_graph_structure = complete_graph_structure
        self.M1 = nn.Parameter(torch.FloatTensor(size=(self.args.num_node, 20)), requires_grad=True)
        self.M2 = nn.Parameter(torch.FloatTensor(size=(self.args.num_node, 20)), requires_grad=True)
        self.k = int(self.args.retain_ratio * self.args.num_node * self.args.num_node)
        self._reset_parameters()

    def _reset_parameters(self):
        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_uniform_(self.M1, gain=gain)
        nn.init.xavier_uniform_(self.M2, gain=gain)

    def forward(self, x):
        graph = copy.deepcopy(self.complete_graph_structure)
        adj_mx = F.relu(torch.matmul(self.M1, self.M2.transpose(0, 1)))
        adj_mx = F.sigmoid(adj_mx)
        edges_w = adj_mx.view(self.args.num_node * self.args.num_node)
        k = torch.topk(edges_w, self.k)[0].min()
        mask = edges_w < k
        graph.remove_edges(mask.nonzero().squeeze())
        graph.edata['w'] = edges_w[~mask]
        return graph
