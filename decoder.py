import numpy as np
import torch
import torch.nn as nn
import torchdiffeq
from gat import GATDecoder as GAT


class Decoder(nn.Module):
    def __init__(self, args):
        super(Decoder, self).__init__()
        self.args = args
        if self.args.decoder_interval == None:
            self.T = torch.linspace(0., self.args.seq_out, 1 + self.args.seq_out) * self.args.decoder_scale
        else:
            self.T = torch.linspace(0., self.args.seq_out,
                                    self.args.decoder_interval * self.args.seq_out + 1) * self.args.decoder_scale
            self.id_train = list(
                np.arange(self.args.decoder_interval, self.args.decoder_interval * (self.args.seq_out + 1),
                          self.args.decoder_interval))

        self.ode_func = GAT(args=self.args, in_dim=self.args.hidden_dim,
                            out_dim=self.args.hidden_dim, num_layers=self.args.num_layers,
                            dropout=self.args.dropout, num_heads=self.args.num_heads)
        self.ode_dynamics = ODEDynamic(ode_func=self.ode_func, rtol=self.args.decoder_rtol,
                                       atol=self.args.decoder_atol,
                                       adjoint=self.args.decoder_adjoint, method=self.args.decoder_integrate_mathod)
        self.output_layer = nn.Sequential(nn.Linear(self.args.hidden_dim, self.args.hidden_dim, bias=True),
                                          nn.Tanh(),
                                          nn.Linear(self.args.hidden_dim, self.args.out_dim, bias=True))

    def forward(self, y0, graph):
        self.ode_func.set_graph(graph)
        y0 = y0.squeeze(1)
        out = self.ode_dynamics(self.T, y0).transpose(0, 1)
        out = self.output_layer(out)
        if self.args.decoder_interval == None:
            return out, None
        else:
            return out[:, 1:, :, :], out[:, self.id_train, :, :]


class ODEDynamic(nn.Module):
    def __init__(self, ode_func, rtol=.01, atol=.001, method='dopri5', adjoint=False, terminal=False):
        super(ODEDynamic, self).__init__()
        self.ode_func = ode_func
        self.rtol = rtol
        self.atol = atol
        self.method = method
        self.adjoint = adjoint
        self.terminal = terminal
        self.perform_num = 0

    def forward(self, vt, y0):
        self.perform_num += 1
        integration_time_vector = vt.type_as(y0)
        if self.adjoint:
            out = torchdiffeq.odeint_adjoint(func=self.ode_func, y0=y0, t=integration_time_vector, rtol=self.rtol,
                                             atol=self.atol, method=self.method)
        else:
            out = torchdiffeq.odeint(func=self.ode_func, y0=y0, t=integration_time_vector, rtol=self.rtol,
                                     atol=self.atol, method=self.method)
        return out

    def reset(self):
        self.perform_num = 0
