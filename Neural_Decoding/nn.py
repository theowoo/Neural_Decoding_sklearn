try:
    from torch import nn
except ImportError:
    print(
        "\nWARNING: PyTorch package is not installed. You will be unable to use some"
        "neural net decoders"
    )
    pass


class FNN(nn.Module):
    def __init__(self, num_units=10, frac_dropout=0, n_layers=1, n_targets=2):
        super().__init__()

        assert n_layers > 0
        self.n_layers = n_layers

        self.dense = nn.ModuleList()
        for _ in range(n_layers):
            self.dense.append(nn.LazyLinear(num_units))

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(frac_dropout)
        self.output = nn.Linear(num_units, n_targets)

    def forward(self, X, **kwargs):

        for i in range(self.n_layers):
            X = self.dense[i](X)
            X = self.relu(X)
            X = self.dropout(X)

        X = self.output(X)

        return X


class RNN(nn.Module):

    def __init__(self, num_units=10, frac_dropout=0, n_targets=2):
        super().__init__()

        self.num_units = num_units
        self.frac_dropout = frac_dropout

        self.rnn = None
        self.dropout = nn.Dropout(frac_dropout)
        self.flatten = nn.Flatten()
        self.output = nn.LazyLinear(n_targets)

    def forward(self, X, **kwargs):

        if self.rnn is None:
            input_size = X.shape[-1]
            self.rnn = nn.RNN(
                input_size=input_size,
                hidden_size=self.num_units,
                nonlinearity="relu",
                dropout=self.frac_dropout,
            ).to(X.device)

        X, hidden_state = self.rnn(X)
        X = self.dropout(X)
        X = self.flatten(X)
        X = self.output(X)

        return X
