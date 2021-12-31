import dgl
import scipy as sp
import torch.nn as nn
from decoder import Decoder
from encoder import Encoder
from graph_learner import GraphLearner
from scipy import sparse as sp


class Model(nn.Module):
    def __init__(self, args):
        super(Model, self).__init__()
        self.args = args
        self.encoder = Encoder(args=args)
        self.hidden_transformation = nn.Sequential(nn.Linear(self.args.hidden_dim, self.args.hidden_dim, bias=True),
                                                   nn.Tanh(),
                                                   nn.Linear(self.args.hidden_dim, self.args.hidden_dim, bias=True))
        self.decoder = Decoder(args=args)
        if args.graph == "geo":
            self.graph = self._generate_graph().to(self.args.device)
        else:
            complete_graph = self._generate_graph().to(self.args.device)
            self.graph_learner = GraphLearner(self.args, complete_graph)

    def forward(self, x):
        if self.args.graph != "geo":
            graph = self.graph_learner(x)
        else:
            graph = self.graph
        hidden_state = self.encoder(x, graph)
        hidden_state = self.hidden_transformation(hidden_state)
        if self.args.decoder_interval == None:
            out = self.decoder(hidden_state)
            return out
        else:
            continous_out, decrete_out = self.decoder(hidden_state, graph)
            return continous_out, decrete_out

    def _generate_graph(self):
        adj_mx = sp.csr_matrix(self.args.adj_mx)
        graph = dgl.from_scipy(adj_mx, eweight_name='w')
        return graph
