import torch
import torch.utils.data as data
from PIL import Image
import numpy as np
import pandas as pd

class FingerprintDataset(data.Dataset):
    def __init__(self, data: pd.DataFrame, size: tuple[int, int]) -> None:
        super().__init__()
        self.data = data
        self.size = size

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # read in the image
        path = self.data.iloc[idx][0]
        target = self.data.iloc[idx][1]
        img = Image.open(path)
        # Resize and 2D grayscale img to 3D array
        img = np.array(img.resize(self.size))
        img = img[:, :, np.newaxis] 

        # channel-last to channel-first for PyTorch
        img = torch.from_numpy(img.transpose(2, 0, 1))
        return img.float(), torch.tensor(target)

def fingerprint_data_loader(dataset: data.Dataset, batch_size: int = 32, shuffle = True):
    return data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, pin_memory=True, num_workers=2)
