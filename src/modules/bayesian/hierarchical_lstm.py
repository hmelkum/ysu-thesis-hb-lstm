from pyro.nn import PyroModule

from src.modules.bayesian.lstm import BayesianLSTM


class HierarchicalBayesianModel(PyroModule):
    def __init__(self, news_input_dim, return_input_dim, hidden_size=16):
        super().__init__()
        self.model1 = BayesianLSTM("model1", news_input_dim, hidden_size, 1)
        self.model2 = BayesianLSTM("model2", return_input_dim, hidden_size, 1)

    def forward(self, news_input, return_input, return_output=None):
        _ = self.model1(news_input, return_output)

        self.model2.fc.weight = self.model1.fc.weight
        self.model2.fc.bias = self.model1.fc.bias

        return self.model2(return_input, return_output)