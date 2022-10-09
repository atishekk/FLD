from torchvision.models import resnet50, ResNet50_Weights
from torch import nn

class ResNet50(nn.Module):
    def __init__(self) -> None:
        super(ResNet50, self).__init__()
        weights = ResNet50_Weights.DEFAULT
        self.encoder = resnet50(weights)
        self.preprocess = weights.transforms()

    def forward(self, X):
        X = self.preprocess(X)
        return self.encoder(X)

    def get_encoder(self):
        pass
