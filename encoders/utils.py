import torch
from torch import nn, optim
import torch.utils.data as data
from tqdm import tqdm
import gc

def train(model: nn.Module, dataset: data.DataLoader,
        epochs: int,
        loss,
        optimiser: optim.Optimizer):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 
    model = model.train().to(device=device, memory_format=torch.channels_last)
    state = []
    for epoch in range(epochs):
        epoch_loss = 0
        with tqdm(dataset, unit='batch') as tepoch:
            for count, (i, t) in enumerate(tepoch):
                tepoch.set_description(f"Epoch {epoch}")
                i, t = i.to(device, memory_format=torch.channels_last), t.to(device, memory_format=torch.channels_last)
                optimiser.zero_grad()
                outputs = model(i)
                ls = loss(outputs, t)
                ls.backward()
                optimiser.step()

                with torch.no_grad():
                    epoch_loss += ls.item()
                    tepoch.set_postfix(loss=epoch_loss/(count + 1)) 
        state.append(epoch_loss)
    gc.collect()
    return state
