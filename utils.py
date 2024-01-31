import torch
from collections import defaultdict

_OPTIMIZERS = None

def set_global_optimizers(optimizers):
    global _OPTIMIZERS
    _OPTIMIZERS = optimizers
    print("set optimizers!")

def get_global_optimizers():
    global _OPTIMIZERS
    assert _OPTIMIZERS != None
    return _OPTIMIZERS

def reset_global_optimizers():
    global _OPTIMIZERS
    assert _OPTIMIZERS != None
    # for param_group_name in _OPTIMIZERS.optimizers:
    #     _OPTIMIZERS.optimizers[param_group_name].__setstate__({'state': defaultdict(dict)})
    _OPTIMIZERS.optimizers["field_features"].__setstate__({'state': defaultdict(dict)})

def exponential_update(old_value, new_value, update_rate):
    return torch.exp(update_rate*torch.log(old_value + 1e-40) + (1-update_rate)*torch.log(new_value + 1e-40))

def set_requires_grad(module, val):
    for p in module.parameters():
        p.requires_grad = val