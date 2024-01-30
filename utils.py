import torch

def exponential_update(old_value, new_value, update_rate):
    return torch.exp(update_rate*torch.log(old_value + 1e-40) + (1-update_rate)*torch.log(new_value + 1e-40))