import torch
from flexnerf_field import MultiLayerTetra

aabb = torch.tensor([[0,1],[0,1],[0,1]])
multi_layer = MultiLayerTetra(aabb, 32, 20, 800, 100)
'''
a
'''