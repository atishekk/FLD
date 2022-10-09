from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from torch import nn

class EfficientNetB0(nn.Module):
    def __init__(self) -> None:
        super(EfficientNetB0, self).__init__()
        weights = EfficientNet_B0_Weights.DEFAULT
        self.encoder = efficientnet_b0(weights = weights) 
        self.preprocess = weights.transforms()

    def forward(self, X):
        X = self.preprocess(X)
        return self.encoder(X)

    def get_encoder(self):
        pass
