import torch
from collections import defaultdict

_OPTIMIZERS = None
_HIT = 0

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
    _OPTIMIZERS.optimizers["auxilary_field_network"].__setstate__({'state': defaultdict(dict)})

def reset_global_optimizers_patial(cell_num_thresh):
    global _OPTIMIZERS
    assert _OPTIMIZERS != None
    torch.set_printoptions(threshold=128)
    state = _OPTIMIZERS.optimizers["field_features"].__getstate__()['state']
    # somehow hacking. there is only on param, which is a large tensor.
    for param in state.keys():
        state[param]['exp_avg'][cell_num_thresh:, :] = 0
        state[param]['exp_avg_sq'][cell_num_thresh:, :] = 0

def mute_global_optimizer(param_group_name):
    global _OPTIMIZERS
    assert _OPTIMIZERS != None
    _OPTIMIZERS.mute_optimizer(param_group_name)

def clear_mute_global_optimizer():
    global _OPTIMIZERS
    assert _OPTIMIZERS != None
    _OPTIMIZERS.clear_mute_optimizer()

def exponential_update(old_value, new_value, update_rate):
    return torch.exp(update_rate*torch.log(old_value + 1e-40) + (1-update_rate)*torch.log(new_value + 1e-40))

def set_requires_grad(module, val):
    for p in module.parameters():
        p.requires_grad = val

def visualize_importance_linelike(step, direct_result, mid_result):
    base_path = "/home/zhangjw/nerfstudio/outputs/lego_sta/flexnerf/logs"
    import os
    if not os.path.exists(base_path + "/linelike_imgs"):
        os.mkdir(base_path + "/linelike_imgs")
    index = len(os.listdir(base_path + "/linelike_imgs"))
    dir_name = base_path + "/linelike_imgs/" + str(index).zfill(8)
    os.mkdir(dir_name)