import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchdiffeq
from gat import GAT
from scipy import sparse as sp


class Encoder(nn.Module):
    def __init__(self, args):
        super(Encoder, self).__init__()
        self.args = args
        self.linear_in = nn.Linear(self.args.in_dim, self.args.hidden_dim)
        self.T = torch.linspace(0., 1., self.args.encoder_interval + 1) * self.args.encoder_scale
        self.graph = self._generate_graph().to(self.args.device)

        self.ode_func = GAT(args=self.args, in_dim=self.args.hidden_dim, out_dim=self.args.hidden_dim, num_layers=1,
                            dropout=self.args.dropout, num_heads=self.args.num_heads, graph=self.graph)
        self.ode_dynamics = ODEDynamic(ode_func=self.ode_func, rtol=self.args.encoder_rtol, atol=self.args.encoder_atol,
                                       adjoint=self.args.encoder_adjoint, method=self.args.encoder_integrate_mathod)
        self.output_layer = nn.Sequential(nn.Linear(self.args.hidden_dim, self.args.hidden_dim, bias=True),
                                          nn.Tanh(),
                                          nn.Linear(self.args.hidden_dim, self.args.hidden_dim, bias=True))
        self.W = nn.Linear(self.args.hidden_dim, self.args.hidden_dim)
        self.U = nn.Linear(self.args.hidden_dim, self.args.hidden_dim)
        self.layer_norm = nn.LayerNorm([self.args.hidden_dim],
                                       elementwise_affine=False)

    def forward(self, inputs):
        inputs = self.linear_in(inputs)
        x = inputs[:, 0, :, :].contiguous().squeeze(1)
        ret = x
        for idx in range(self.args.seq_in):
            ret = self.ode_dynamics(self.T, ret)[-1]
            if idx != 0:
                input = inputs[:, idx, :, :].contiguous().squeeze(1)
                ret = F.tanh(self.W(ret) + self.U(input))
            ret = self.layer_norm(ret)
        out = ret.view(self.args.batch_size, self.args.num_node, self.args.hidden_dim).unsqueeze(0)
        out = out.transpose(0, 1)
        out = self.output_layer(out)
        return out

    def _generate_graph(self):
        adj_mx = sp.csr_matrix(self.args.adj_mx)
        graph = dgl.from_scipy(adj_mx, eweight_name='w')
        return graph


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
