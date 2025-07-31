from torch.nn import Module, Linear, ReLU


class MultiLinearProbe(Module):
    def __init__(self, in_features, out_features):
        super(MultiLinearProbe, self).__init__()
        self.linear1 = Linear(in_features, in_features // 2)
        self.linear2 = Linear(in_features // 2, out_features)
        self.relu = ReLU()

    def forward(self, x):
        x = self.relu(self.linear1(x))
        return self.linear2(x)