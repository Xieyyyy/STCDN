import torch.nn as nn

from decoder import Decoder
from encoder import Encoder


class Model(nn.Module):
    def __init__(self, args):
        super(Model, self).__init__()
        self.args = args
        self.encoder = Encoder(args=args)
        self.hidden_transformation = nn.Sequential(nn.Linear(self.args.hidden_dim, self.args.hidden_dim, bias=True),
                                                   nn.Tanh(),
                                                   nn.Linear(self.args.hidden_dim, self.args.hidden_dim, bias=True))
        self.decoder = Decoder(args=args)

    def forward(self, x):
        hidden_state = self.encoder(x, self.args.adj_mx)
        hidden_state = self.hidden_transformation(hidden_state)
        if self.args.decoder_interval == None:
            out = self.decoder(hidden_state)
            return out
        else:
            continous_out, decrete_out = self.decoder(hidden_state, self.args.adj_mx)
            return continous_out, decrete_out
