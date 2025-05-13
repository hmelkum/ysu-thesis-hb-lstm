import pyro
import pyro.distributions as dist
from pyro.nn import PyroModule, PyroSample
import torch
import torch.nn as nn

from src.modules.bayesian.lstm_cell import BayesianLSTMCell


class BayesianLSTM(PyroModule):
    def __init__(self, name, input_size, hidden_size, output_size):
        super().__init__()
        self.name = name
        self.lstm_cell = BayesianLSTMCell(input_size, hidden_size)
        self.fc = PyroModule[nn.Linear](hidden_size, output_size)
        self.fc.weight = PyroSample(dist.Normal(0., 1.).expand([output_size, hidden_size]).to_event(2))
        self.fc.bias = PyroSample(dist.Normal(0., 1.).expand([output_size]).to_event(1))

    def forward(self, x, y=None):
        batch_size, seq_len, _ = x.size()
        h = x.new_zeros(batch_size, self.lstm_cell.hidden_size)
        c = x.new_zeros(batch_size, self.lstm_cell.hidden_size)
        for t in range(seq_len):
            h, c = self.lstm_cell(x[:, t, :], (h, c))

        output = self.fc(h).squeeze(-1)

        sigma = pyro.param(f"{self.name}_sigma", torch.tensor(1.0), constraint=dist.constraints.positive)
        
        with pyro.plate(f"{self.name}_data", output.size(0)):
            pyro.sample(f"{self.name}_obs", dist.Normal(output, sigma), obs=y)

        return output