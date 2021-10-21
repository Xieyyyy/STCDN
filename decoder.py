import dgl
import numpy as np
import torch
import torch.nn as nn
import torchdiffeq
from scipy import sparse as sp

from gat import GAT


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

        self.graph = self._generate_graph()
        self.ode_func = GAT(args=self.args, in_dim=self.args.hidden_dim, out_dim=self.args.hidden_dim, num_layers=1,
                            dropout=self.args.dropout, num_heads=self.args.num_heads, graph=self.graph)
        self.ode_dynamics = ODEDynamic(ode_func=self.ode_func, rtol=self.args.decoder_rtol,
                                       atol=self.args.decoder_atol,
                                       adjoint=self.args.decoder_adjoint, method=self.args.decoder_integrate_mathod)
        self.output_layer = nn.Sequential(nn.Linear(self.args.hidden_dim, self.args.hidden_dim, bias=True),
                                          nn.Tanh(),
                                          nn.Linear(self.args.hidden_dim, self.args.out_dim, bias=True))

    def forward(self, y0):
        y0 = y0.view(self.args.batch_size * self.args.num_node, self.args.hidden_dim)
        out = self.ode_dynamics(self.T, y0)
        if self.args.decoder_interval == None:
            out = out.view(self.args.seq_out + 1, self.args.batch_size, self.args.num_node,
                           self.args.hidden_dim)
        else:
            out = out.view(self.args.decoder_interval * self.args.seq_out + 1, self.args.batch_size, self.args.num_node,
                           self.args.hidden_dim)
        out = out.transpose(0, 1)
        out = self.output_layer(out)
        if self.args.decoder_interval == None:
            return out, None
        else:
            return out[:, 1:, :, :], out[:, self.id_train, :, :]

    def _generate_graph(self):
        adj_mx = sp.csr_matrix(self.args.adj_mx)
        graph = dgl.from_scipy(adj_mx, eweight_name='w')
        batch_graph = dgl.batch([graph for _ in range(self.args.batch_size)])
        return batch_graph.to(self.args.device)


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
