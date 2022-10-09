from torch import nn

class ANN(nn.Module):
    def __init__(self, in_features: int):
        super(ANN, self).__init__()
        self.flatten = nn.Flatten()
        self.classifier = nn.Sequential(
                nn.Linear(in_features, 512),
                nn.ReLU(),
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.Linear(256, 1),
                nn.Softmax()
        )

    def forward(self, X):
        X = self.flatten(X)
        return self.classifier(X)

    def fit(self, X, y):
        pass
