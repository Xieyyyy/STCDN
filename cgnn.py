import torch
import torch.nn as nn
import torch.nn.functional as F
import torchdiffeq


class CGNNBlock(nn.Module):
    def __init__(self, args):
        super(CGNNBlock, self).__init__()
        self.args = args
        self.adj_mx = None
        self.x0 = None
        self.alpha = self.args.alpha
        self.alpha_train = nn.Parameter(self.alpha * torch.ones(self.args.num_node))
        self.w = nn.Parameter(torch.eye(self.args.hidden_dim))
        self.d = nn.Parameter(torch.ones(self.args.hidden_dim))
        self.nfe = 0

    def forward(self, vt, x):
        self.nfe += 1
        alph = F.sigmoid(self.alpha_train).unsqueeze(dim=-1)  # [170,1]
        ax = torch.matmul(self.adj_mx, x)  # [64,170,128]
        d = torch.clamp(self.d, min=0, max=1)
        w = torch.mm(self.w * d, torch.t(self.w))
        xw = torch.matmul(x, w)
        f = alph * 0.5 * (ax - x) + xw - x + self.x0
        return f

    def set_adj_mx(self, adj_mx):
        self.adj_mx = adj_mx

    def set_x0(self, x0):
        self.x0 = x0.clone().detach()

    def reset_nfe(self):
        self.nfe = 0


class ODEDynamics(nn.Module):
    def __init__(self, ode_func, rtol=.01, atol=.001, method='euler', adjoint=False, T=False):
        super(ODEDynamics, self).__init__()
        self.T = T
        self.ode_func = ode_func
        self.rtol = rtol
        self.atol = atol
        self.method = method
        self.adjoint = adjoint

    def set_x0(self, x0):
        self.odefunc.x0 = x0.clone().detach()

    def forward(self, vt, x):
        t = self.T.type_as(x)
        if self.adjoint:
            out = torchdiffeq.odeint_adjoint(func=self.ode_func, y0=x, t=t, rtol=self.rtol,
                                             atol=self.atol, method=self.method)
        else:
            out = torchdiffeq.odeint(func=self.ode_func, y0=x, t=t, rtol=self.rtol,
                                     atol=self.atol, method=self.method)
        return out


class CGNN(nn.Module):
    def __init__(self, args):
        super(CGNN, self).__init__()
        self.args = args
        self.adj_mx = None
        self.T = torch.linspace(0., 1,
                                self.args.spatial_interval + 1) * self.args.spatial_scale
        self.cgnn_block = CGNNBlock(args)
        self.ode_dynamics = ODEDynamics(ode_func=self.cgnn_block, method=self.args.spatial_integrate_mathod,
                                        adjoint=self.args.spatial_adjoint, T=self.T)
        self.nfe = 0
        self.fc_in = nn.Linear(self.args.hidden_dim, self.args.hidden_dim)
        self.fc_out = nn.Linear(self.args.hidden_dim, self.args.hidden_dim)

    def forward(self, vt, x):
        self.nfe += 1
        self.cgnn_block.set_x0(x)
        self.cgnn_block.set_adj_mx(self.adj_mx)
        x = self.fc_in(x)
        x = F.dropout(x, self.args.dropout, training=self.training)
        out = self.ode_dynamics(self.T, x)[-1]
        self.cgnn_block.reset_nfe()
        return out

    def reset(self):
        self.nfe = 0

    def set_adj_mx(self, adj_mx):
        self.adj_mx = adj_mx
