import torch
from flexnerf_field import MultiLayerTetra

MANUAL_TEST = True
TEST_ONE_SEED = False
SCAN_SEEDS = False

if MANUAL_TEST:
  torch.manual_seed(122)
  aabb = torch.tensor([[0,1],[0,1],[0,1]])
  multi_layer = MultiLayerTetra(aabb, 1, 20, 16, 10)
  with torch.no_grad():
    multi_layer.field[:4,:] = torch.tensor([1,0,0,0]).view(4,1)
  xyz = torch.zeros(4,3).uniform_(0,1)*0.5 + 0.25
  print(multi_layer.forward(xyz, use_extend=False))
  print("\n")
  print(multi_layer.field[:5, :])
  print("\n")
  print(xyz)

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