import numpy as np
from .descriptor import SIFTDescriptor

class MinutiaeDetection:
    def __init__(self, descriptor = SIFTDescriptor()) -> None:
        self.descriptor = descriptor

    def point_detection(self) -> np.ndarray:
        raise NotImplemented("TODO")
