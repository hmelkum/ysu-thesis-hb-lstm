import pyro.distributions as dist
from pyro.nn import PyroModule, PyroSample
import torch


class BayesianLSTMCell(PyroModule):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.ih = PyroSample(dist.Normal(0., 1.).expand([4 * hidden_size, input_size]).to_event(2))
        self.hh = PyroSample(dist.Normal(0., 1.).expand([4 * hidden_size, hidden_size]).to_event(2))
        self.bias = PyroSample(dist.Normal(0., 1.).expand([4 * hidden_size]).to_event(1))

    def forward(self, x, hc):
        h, c = hc
        gates = torch.matmul(x, self.ih.transpose(-2, -1)) + torch.matmul(h, self.hh.transpose(-2, -1)) + self.bias
        i, f, g, o = gates.chunk(4, 1)
        i, f, o = torch.sigmoid(i), torch.sigmoid(f), torch.sigmoid(o)
        g = torch.tanh(g)
        c_new = f * c + i * g
        h_new = o * torch.tanh(c_new)
        return h_new, c_new
