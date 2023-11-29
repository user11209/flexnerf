import torch
from flexnerf_field import MultiLayerTetra

TEST_ONE_SEED = True
SCAN_SEEDS = False

if TEST_ONE_SEED:
  torch.manual_seed(97)
  aabb = torch.tensor([[0,1],[0,1],[0,1]])
  multi_layer = MultiLayerTetra(aabb, 32, 20, 16, 10)

  print("++++++++++++++ REMOVING ++++++++++++++")
  multi_layer.remove_last_sampled()
  print("============= POINT ============")
  print(multi_layer.point_index)
  print("============== EDGE ============")
  print(multi_layer.edge_index)
  print("++++++++++++++ NEW SAMPLE +++++++++++++++")
  # edge_valid = multi_layer.sample_extend()
  # multi_layer.apply_sampled(edge_valid)
  multi_layer.merge_extend(torch.tensor(2, dtype=torch.int64))
  print(multi_layer.edge_offset[0])
  print("============= POINT ============")
  print(multi_layer.point_index)
  print("============== EDGE ============")
  print(multi_layer.edge_index)

  print("++++++++++++++ NEW SAMPLE +++++++++++++++")
  multi_layer.merge_extend(torch.tensor(3, dtype=torch.int64))
  print(multi_layer.edge_offset[0])
  print("============= POINT ============")
  print(multi_layer.point_index)
  print("============== EDGE ============")
  print(multi_layer.edge_index)

if SCAN_SEEDS:
  for i in range(100):
    torch.manual_seed(i)
    print("!!!!!!!!!!!!!!!!!!!!!!!!! seed ", i, " !!!!!!!!!!!!!!!!!!!!!!!!!!")
    aabb = torch.tensor([[0,1],[0,1],[0,1]])
    multi_layer = MultiLayerTetra(aabb, 32, 20, 16, 10)
    multi_layer.importance.uniform_(0,1)
    multi_layer.remove_last_sampled()
    multi_layer.merge_extend(torch.tensor(5, dtype=torch.int64))
    multi_layer.merge_extend(torch.tensor(7, dtype=torch.int64))
    if False:
      print(multi_layer.edge_offset[0])
      print("============= POINT ============")
      print(multi_layer.point_index)
      print("============== EDGE ============")
      print(multi_layer.edge_index)
'''
a
'''