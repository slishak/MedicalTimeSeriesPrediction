import torch

device = torch.device('cpu')

def switch_device(to: str):
    global device
    device = torch.device(to)
