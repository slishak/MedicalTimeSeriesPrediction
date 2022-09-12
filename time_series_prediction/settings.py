import torch

device = torch.device('cpu')


def switch_device(to: str):
    """Switch global PyTorch device setting

    Args:
        to (str): 'cpu' or 'cuda'
    """
    global device
    device = torch.device(to)
