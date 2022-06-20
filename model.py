import dgl
import numpy as np
import scipy as sp
import torch
import torch.nn as nn
from scipy import sparse as sp

from decoder import Decoder
from encoder import Encoder
from graph_learner import GraphLearner


class Model(nn.Module):
    def __init__(self, args):
        super(Model, self).__init__()
        self.args = args
        self.encoder = Encoder(args=args)
        self.hidden_transformation = Variation(args)
        self.decoder = Decoder(args=args)
        if args.graph == "geo":
            self.graph = self._generate_graph().to(self.args.device)
        else:
            self.graph_learner = GraphLearner(self.args)

    def forward(self, x):
        if self.args.graph != "geo":
            graph = self.graph_learner(x)
        else:
            graph = self.graph
        hidden_state = self.encoder(x, graph)
        hidden_state, mu, sigma = self.hidden_transformation(hidden_state)

        decrete_out = self.decoder(hidden_state, graph)
        return decrete_out, mu, sigma

    def _generate_graph(self):
        adj_mx = sp.csr_matrix(self.args.adj_mx)
        graph = dgl.from_scipy(adj_mx, eweight_name='w')
        return graph


class Variation(nn.Module):
    def __init__(self, args):
        super(Variation, self).__init__()
        self.args = args
        self.enc_mu = nn.Sequential(nn.Linear(self.args.hidden_dim, self.args.hidden_dim, bias=True),
                                    nn.Tanh(),
                                    nn.Linear(self.args.hidden_dim, self.args.hidden_dim, bias=True))
        self.enc_sigma = nn.Sequential(nn.Linear(self.args.hidden_dim, self.args.hidden_dim, bias=True),
                                       nn.Tanh(),
                                       nn.Linear(self.args.hidden_dim, self.args.hidden_dim, bias=True))

    def forward(self, x):
        mu = self.enc_mu(x)
        log_sigma = self.enc_sigma(x)
        sigma = torch.exp(log_sigma)
        std_z = torch.from_numpy(np.random.normal(0, 1, size=sigma.size())).float().to(mu.device)
        return mu + sigma * nn.Parameter(std_z, requires_grad=False), mu, sigma
