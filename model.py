import torch
import torch.nn as nn

from decoder import Decoder
from encoder import Encoder
from graph_learner import GraphLearner


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
            self.adj = torch.Tensor(self.args.adj_mx).to(self.args.device)
        else:
            self.graph_learner = GraphLearner(self.args)

    def forward(self, x):
        if self.args.graph != "geo":
            adj_mx = self.graph_learner(x)
        else:
            adj_mx = self.adj
        hidden_state = self.encoder(x, adj_mx)
        hidden_state = self.hidden_transformation(hidden_state)
        if self.args.decoder_interval == None:
            out = self.decoder(hidden_state)
            return out
        else:
            continous_out, decrete_out = self.decoder(hidden_state, adj_mx)
            return continous_out, decrete_out
