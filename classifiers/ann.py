import torch
from torch import nn, optim

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

    def fit(self, X, y, epochs, batch_size=32, optimiser=optim.Adam, loss = nn.CrossEntropyLoss(), lr = 1e-6):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self = self.to(device, memory_format=torch.channels_last)
        optim = optimiser(self.parameters(), lr=lr)
        state = []

        for epoch in range(epochs):
            epoch_loss = 0
            for idx in range(0, len(X), batch_size):
                batch_x, batch_y = X[idx: idx+batch_size], y[idx: idx+batch_size]
                batch_x, batch_y = batch_x.to(device, memory_format=torch.channels_last), batch_y.to(device, memory_format=torch.channels_last)
                optim.zero_grad()
                pred = self(batch_x)
                ls = loss(pred, batch_y)
                ls.backward()
                optim.step()
                with torch.no_grad():
                    epoch_loss += ls.item()

            state += [epoch_loss / len(X)]
        return state

