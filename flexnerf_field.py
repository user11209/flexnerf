from typing import Dict, Literal, Optional, Tuple

import torch
from torch import Tensor, nn

from nerfstudio.cameras.rays import RaySamples
from nerfstudio.data.scene_box import SceneBox
from nerfstudio.field_components.activations import trunc_exp
from nerfstudio.field_components.embedding import Embedding
from nerfstudio.field_components.encodings import HashEncoding, NeRFEncoding, SHEncoding
from nerfstudio.field_components.field_heads import (
    FieldHeadNames,
    PredNormalsFieldHead,
    SemanticFieldHead,
    TransientDensityFieldHead,
    TransientRGBFieldHead,
    UncertaintyFieldHead,
)
from nerfstudio.field_components.mlp import MLP
from nerfstudio.field_components.spatial_distortions import SpatialDistortion
from nerfstudio.fields.base_field import Field, get_normalized_directions

import flexnerf.utils as utils
from .utils import exponential_update, set_requires_grad, visualize_importance_linelike

DIY = False
DEBUG = True
DEBUG_file = "/home/zhangjw/nerfstudio/external/flexnerf/log_file.txt"
DEBUG_over_file = "/home/zhangjw/nerfstudio/external/flexnerf/over_log_file.txt"
DEBUG_FLAG = 0
DEBUG_FLAG_INNER = 0

FILTER_OUT_BOX = True

def to_canonical(xyz, cell_xyz):
    xyz: Tensor[sample_num, 3]
    cell_xyz: Tensor[sample_num, 4, 3]
    canonical: Tensor[sample_num, 4]

    # [a, b, c, d] @ convert_matrix = [x, y, z, 1]
    sample_num = cell_xyz.shape[0]
    cell_xyz_tail = torch.ones(sample_num, 4, 1, dtype=cell_xyz.dtype, device=cell_xyz.device)
    convert_matrix = torch.transpose( torch.concatenate([cell_xyz, cell_xyz_tail], axis=-1), 1,2)
    convert_xyz = torch.concatenate([xyz, torch.ones_like(xyz[:,:1])], axis=-1)
    canonical = torch.linalg.solve(convert_matrix, convert_xyz)
    return canonical

def interpolate_field_value(xyz, cells):
    xyz: Tensor[sample_num, each_cell_num, 3] or Tensor[sample_num, 3]
    cell_xyz: Tensor[sample_num, 4, 3]

    if len(xyz.shape) == 3:
        xyz = xyz.view(-1,3)
        cell_xyz = cells["xyz"].unsqueeze(1).expand(-1, xyz.shape[1], -1, -1).view(-1, 4, 3)
        cell_feature = cells["feature"].unsqueeze(1).expand(-1, xyz.shape[1], -1, -1).view(xyz.shape[0]*xyz.shape[1], 4, -1)
    else:
        cell_xyz = cells["xyz"]
        cell_feature = cells["feature"]

    canonical = to_canonical(xyz, cell_xyz).unsqueeze(-1)
    with open(DEBUG_over_file, "w") as write_file:
        write_file.write("[interpolate_field_value]: "+str(cell_feature.shape)+" "+str(canonical.shape))
    interpolated_feature = torch.sum(cell_feature * canonical, axis=1)
    import numpy as np
    torch.set_printoptions(threshold=np.inf)
    if False:
        with open(DEBUG_over_file, "a") as write_file:
            write_file.write("[interpolate_field_value]: "+str(interpolated_feature.shape)+"\n")
            write_file.write("========================= four features =========================\n")
            write_file.write(str(cell_feature[::8000,:,:]))
            write_file.write("\n=================================================================\n")
            write_file.write(str(canonical[::8000, :, :]))
    if len(xyz.shape) == 3:
        interpolated_feature = interpolated_feature.view(xyz.shape[0], xyz.shape[1], -1)
    return interpolated_feature

def divide_into_children(xyz, cells, printing=False):
    xyz: Tensor[sample_num, each_cell_num, 3] or Tensor[sample_num, 3]
    cell_xyz: Tensor[sample_num, 8, 3]
    cell_cut: Tensor[sample_num, 2]

    if len(xyz.shape) == 3:
        xyz = xyz.view(-1,3)
        cell_xyz = cells["xyz"].unsqueeze(1).expand(-1, xyz.shape[1], -1, -1).view(-1, 4, 3)
        cell_cut = cells["cut"].unsqueeze(1).expand(-1, xyz.shape[1], -1).view(xyz.shape[0]*xyz.shape[1], -1)
    else:
        cell_xyz = cells["xyz"]
        cell_cut = cells["cut"]

    cell_cut = cell_cut % 4

    canonical = to_canonical(xyz, cell_xyz)

    canonical_cut = torch.gather(canonical, 1, cell_cut.long())
    children_choice = (canonical_cut[:,0] - canonical_cut[:,1] < 0).long()
    # return shape of (sample_num)
    return children_choice

class MultiLayerTetra(nn.Module):
    def __init__(self,
                aabb,
                feature_dim = 32,
                max_layer_num=20,
                max_cell_count=8e4,
                max_point_count=1e4):
        super().__init__()
        torch.manual_seed(1423457)
        self.step = 0
        # this list is hard to be reorganized, if using model.load, this list will not reorganize itself, but only the underlying parameters and buffers
        self.register_buffer("aabb", aabb)
        self.register_buffer("feature_dim", torch.tensor(feature_dim, dtype=torch.int32))
        self.register_buffer("max_layer_num", torch.tensor(max_layer_num, dtype=torch.int32))
        self.register_buffer("max_cell_count", torch.tensor(max_cell_count, dtype=torch.int32))
        self.register_buffer("max_edge_count", torch.tensor(2*max_cell_count+5, dtype=torch.int32))
        self.register_buffer("max_point_count", torch.tensor(max_point_count, dtype=torch.int32))
        self.register_buffer("first_layer_cell_num", torch.tensor(1, dtype=torch.int32))
        assert 3*self.first_layer_cell_num <= max_cell_count, "Variable `max_cell_count` is too small, it at least need to contain all cells on the second layer"

        # !note: cells or points are continuously stored, thus they span a range of cell_offset[i]~cell_offset[i+1]
        #? this is currently not obeyed. all cells and points are not sorted, and we use cell_offset[-1] to find the range
        # **original** cells or points are continuously stored
        self.register_buffer("cell_offset", torch.zeros(2, dtype=torch.int64))
        self.register_buffer("edge_offset", torch.zeros(1, dtype=torch.int64))
        self.register_buffer("point_offset", torch.zeros(2, dtype=torch.int64))
        # **extend** cells or points are continuously stored, 
        # note that the first a few layers are fully covered by original, so extend_offset[i] == extend_offset[i+1] for them
        # extend_offset[0] == 0, and it actually starts from original_offset[-1]
        self.register_buffer("extend_cell_offset", torch.zeros(1, dtype=torch.int64))
        self.register_buffer("extend_point_offset", torch.zeros(1, dtype=torch.int64))

        # !note: the following values contain the important properties of the cell pool
        self.register_buffer(
            "child_index",
            -1*torch.ones((self.max_cell_count, 2), dtype=torch.int64)
        )
        self.register_buffer(
            "edge_index",
            -torch.ones(self.max_cell_count, 6, dtype=torch.int64)
        )
        self.register_buffer(
            "point_index",
            -torch.ones((self.max_cell_count, 4), dtype=torch.int64)
        )
        ## child_cut is the way it is sliced into current cell(1 of 6 edges, or 2 of 4 vertices)
        self.register_buffer(
            "child_cut",
            -torch.ones((self.max_cell_count, 2), dtype=torch.int32)
        )
        self.child_cut[:,1] = 1
        self.register_buffer(
            "parent_index",
            torch.zeros((self.max_cell_count), dtype=torch.int64)
        )
        ## note: this is the importance of each subdivision candidate. If you want to get the importance of a cell subdivision, one way is to set the importance to be the edge of parent cell's cut (perhaps compare to that without the subdivision afterwards). It is necessary to set this to random small values! or else merge_extend will fail to distinguish which edges are leaf edges which need to be sampled on.
        self.register_buffer(
            "inter_level_importance",
            torch.empty((self.max_cell_count, 6), dtype=torch.float32).uniform_(0,1e-8)
        )
        self.register_buffer(
            "remove_empty_importance",
            torch.empty((self.max_cell_count, 6), dtype=torch.float32).uniform_(0,1e-8)
        )
        ## activation_layer stands for the layer where child cell is stored. That is, cells may have a cross-multi-layer parent-child relationship. 
        ## note: A cell is only activated when seach_layer_i(i == activation_layer)
        ## activation_layer in range(0, max_layer_num) for original; range(max_layer_num, 2*max_layer_num) for extend
        self.register_buffer(
            "activation_layer",
            -torch.ones(self.max_cell_count, dtype=torch.int32)
        )
        ## layer is where the cell is stored, in another way, `parent.activation_layer == child.layer`
        self.register_buffer(
            "layer",
            -torch.ones(self.max_cell_count, dtype=torch.int32)
        )
        ## neighbor means the face neighbor of the cell, each cell have 4 neighbor.
        self.register_buffer(
            "neighbor",
            -torch.ones(2, self.max_cell_count, 4, dtype=torch.int32)
        )

        # !note: the following values contain the important properties of the edge pool
        self.register_buffer(
            "edge_importance",
            torch.zeros(self.max_edge_count, dtype=torch.float32)
        )

        # !note: the following values contain the important properties of the point pool
        self.register_parameter(
            "field",
            nn.Parameter(
                torch.zeros((self.max_point_count, self.feature_dim), dtype=torch.float32)
            ),
        )
        self.register_buffer(
            "xyz",
            torch.zeros((self.max_point_count, 3), dtype=torch.float32)
        )
        self.register_buffer(
            "parent_points",
            -torch.ones((self.max_point_count, 2), dtype=torch.int64)
        )

        # !note: the following values contain some flags valueable in tree organization
        self.register_buffer("edge_sorted", torch.tensor([False]))
        self.register_buffer("importance_momentum", torch.tensor([1.0]))
        self.register_buffer("weights_momentum", torch.tensor([1.0]))

        # !note: registering stop here, organize the first layer of tree and try sampling
        self.organize_layer_0()
        self.refine_samples(0, update_original=True, start_step=0, initializing=True)
        self.sorted_edge_offset = None

    @torch.inference_mode()
    def refine_samples(self, step, update_original=True, use_inter_level_importance=True, start_step=0, initializing=False, interval=1000):
        if step != 0 and step < start_step or step > 50000:
            return
        use_inter_level_importance = (step//interval%2 == 0)
        self.importance = self.inter_level_importance if use_inter_level_importance else self.remove_empty_importance
        self.remove_last_sampled(initializing)
        # TODO: check boundaries, no to exceed the `max_??_count`, especially for sample_extend. Note, merge_extend lack 1 positional argument for that reason. temporarily set it to be 2. Note, this value means how many parent cells will be divided, so the actual increased cell number is at most twice the value
        if update_original:
            self.apply_newest_layer_field()
            self.merge_extend(torch.tensor(16, dtype=torch.int64))
            if not initializing:
                utils.reset_global_optimizers()
        edge_valid = self.sample_extend()
        self.apply_sampled(edge_valid)
        if update_original and DEBUG:
            with open(DEBUG_file, "a") as write_file:
                write_file.write("updating: cell "+str(self.cell_offset[0].item())+" edge "+str(self.edge_offset[0].item())+" point "+str(self.point_offset[0].item())+".\n")

    def forward(self, xyz, use_extend=True):
        cells = self.search_layer_0(xyz)
        old_cell_id = cells["cell_id"].clone()
        for i in range(1, self.max_layer_num-1):
            #* this last_layer_cell_id / next_layer_cell_id / old_cell_id thing is to seperate the newly merged cells and the old original cells. it might be important for loss stability.
            last_layer_cell_id = cells["cell_id"]

            cells = self.search_layer_i(xyz, cells, i)

            next_layer_cell_id = cells["cell_id"]
            old_cell_id.masked_scatter(next_layer_cell_id >= self.cell_offset[1], last_layer_cell_id)
        cell_id = cells["cell_id"]
        self.gather_info_from_cell_id(cell_id, cells, point_id_thresh=self.point_offset[1])
        additional_feature = interpolate_field_value(xyz, cells)

        old_cell_id.masked_scatter(old_cell_id == 0, cell_id)
        old_cells = {}
        self.gather_info_from_cell_id(old_cell_id, old_cells)
        old_feature = interpolate_field_value(xyz, old_cells)

        feature = old_feature + additional_feature

        importance = torch.max(self.inter_level_importance[cell_id, :], axis=-1).values
        if not use_extend:
            return feature, None, cell_id, importance
        else:
            cells = self.search_extend(xyz, cells)
            extend_cell_id = cells["cell_id"]
            self.gather_info_from_cell_id(extend_cell_id, cells, point_id_thresh=self.point_offset[0])
            additional_extend_feature = interpolate_field_value(xyz, cells)
            extend_feature = additional_extend_feature + feature.detach()
            #? possibly not necessary to pass all the three values all the time?
            return feature, extend_feature, old_feature, cell_id, extend_cell_id, importance

    def gather_info_from_cell_id(self, cell_id, cells_dict, point_id_thresh=None):
        if cell_id == None:
            cell_id = cells_dict["cell_id"]
        elif "cell_id" not in cells_dict:
            cells_dict["cell_id"] = cell_id
        point_id = self.point_index[cell_id, :]
        if "xyz" not in cells_dict:
            cells_dict["xyz"] = self.xyz[point_id, :]
        if "feature" not in cells_dict:
            cells_dict["feature"] = self.field[point_id, :]
        if "cut" not in cells_dict:
            cells_dict["cut"] = self.child_cut[cell_id, :]
        #* a new feature, blocking gradients from passing to original points. Only gradients of extend points can be utilized.
        if point_id_thresh != None:
            cells_dict["feature"] = torch.where((point_id>=point_id_thresh)[:,:,None], cells_dict["feature"], 0)
        return

    # @torch.inference_mode()
    # def reset_importance(self):
    #     count = self.cell_offset[0]
    #     self.importance[:count, :] = torch.zeros_like(self.importance[:count, :]).uniform_(0,0.00001)

    @torch.inference_mode()
    def global_importance_annealing_callback(self, step, start_step=0):
        if step < start_step:
            return
        #? global importance annealing may have other period, and other ratio
        anneal_round = step // 250
        valid_cell_count = self.cell_offset[0]
        anneal_flag = (anneal_round % torch.pow(2, self.activation_layer[:valid_cell_count]) == 0)
        self.inter_level_importance[:valid_cell_count, :] = torch.where(anneal_flag[:,None].expand(-1,6), 
                                                            0.8*self.inter_level_importance[:valid_cell_count, :],
                                                            self.inter_level_importance[:valid_cell_count, :])


    @torch.inference_mode()
    def update_importance_from_result(self, result, step, start_step):
        '''
        result = {'cell_id':..., 'extend_cell_id':...,
                  'weights_original':..., 'rgb_original':..., 
                  'weights_extend':..., 'rgb_extend':..., 
                  'pred_rgb':..., 'gt_rgb':...}
        '''

        cell_index = result['cell_id']
        extend_cell_index = result['extend_cell_id']
        weights_original = result['weights_original']
        rgb_original = result['rgb_original']
        weights_extend = result['weights_extend']
        rgb_extend = result['rgb_extend']
        pred_rgb = result['pred_rgb']
        gt_rgb = result['gt_rgb']
        transmittance = result['transmittance'].squeeze(-1)

        update_weights_momentum = torch.max(weights_original) + torch.max(weights_extend) + 1e-8
        self.weights_momentum[0] = exponential_update(self.weights_momentum[0], update_weights_momentum, 0.99)


        raw_rgb_ratio_diff = rgb_extend - rgb_original
        raw_rgb_ratio_diff_2 = torch.sum(raw_rgb_ratio_diff*raw_rgb_ratio_diff, axis=-1)
        raw_wei_ratio_diff = (weights_extend - weights_original)/self.weights_momentum[0]
        raw_wei_ratio_diff_2 = torch.abs(raw_wei_ratio_diff*raw_wei_ratio_diff).squeeze(-1)
        extend_diff_2 = raw_rgb_ratio_diff_2 * raw_wei_ratio_diff_2

        acc_rgb_diff_2 = torch.sum((pred_rgb-gt_rgb)*(pred_rgb-gt_rgb), axis=-1)[:, None].expand(extend_diff_2.shape)

        transmittance_thresh = torch.sin(torch.tensor(12.345678)*step)*0.2 +0.8
        raw_transmit_ratio = (transmittance > transmittance_thresh).float()

        importance_ingredients = {"acc_rgb_diff_2":acc_rgb_diff_2, "extend_diff_2":extend_diff_2, 
                                    "extend_cell_index":extend_cell_index, "raw_transmit_ratio":raw_transmit_ratio}

        visualize_importance_linelike(step, result, importance_ingredients)

        reordered_cell_index, reordered_inter_level_importance_value, reordered_importance_power, \
                              reordered_remove_empty_importance_value, reordered_importance_count \
                    = self.calculate_importance_with_reordering(cell_index, importance_ingredients)
        
        # if not started yet, only update importance_momentum. May cause /0 error if not.
        # (because importance_momentum will be 0 for a relatively long time)
        only_momentum = (step < start_step)
        if only_momentum:
            return

        self.update_importance(reordered_cell_index, reordered_inter_level_importance_value, reordered_importance_power, update_inter_level_importance=True)
        self.update_importance(reordered_cell_index, reordered_remove_empty_importance_value, reordered_importance_count,update_inter_level_importance=False)

    @torch.inference_mode()
    def calculate_importance_with_reordering(self, related_cell_index, ingredients, momentum_update_rate=0.998):
        related_cell_index = related_cell_index.view(-1)
        for ingredients_key in ingredients:
            ingredients[ingredients_key] = ingredients[ingredients_key].reshape(-1)
        cell_count = related_cell_index.shape[0]

        #* remove duplicates
        sorted_related_cell_index, sorted_indices = torch.sort(related_cell_index)

        for ingredients_key in ingredients:
            ingredients[ingredients_key] = ingredients[ingredients_key][sorted_indices]

        cell_changing_nohead = (sorted_related_cell_index[1:] - sorted_related_cell_index[:-1] != 0).long()
        head = torch.tensor([1], dtype=torch.int64, device=cell_changing_nohead.device)
        cell_changing = torch.concatenate([head, cell_changing_nohead])

        leading_cell = torch.concatenate([torch.nonzero(cell_changing).view(-1), cell_count*head])
        float_head = torch.tensor([0], dtype=torch.float32, device=cell_changing_nohead.device)
        
        reordered_cell_index = sorted_related_cell_index[leading_cell[:-1]]

        #! calculate inter_level_importance new value.
        sorted_importance_value = torch.sqrt(ingredients["acc_rgb_diff_2"] * ingredients["extend_diff_2"])
        #* update importance momentum
        importance_max = torch.max(sorted_importance_value)
        self.importance_momentum[0] = exponential_update(self.importance_momentum, importance_max, momentum_update_rate)
        #* note, importance_value must be divided here. or else, update_rate in update_importance func will be too close to 1.
        sorted_importance_value = sorted_importance_value / self.importance_momentum
        #* the average below should be a powered weight aware of importance.
        # for example, if importance to update is too small, it should not count! it might be a surface cell, just obstacled in some camera views. for unimportant cells, just wait for the periodic global annealing to remove it.
        sorted_importance_power = sorted_importance_value/(1+sorted_importance_value)
        sorted_power_cumsum = torch.concatenate([float_head, torch.cumsum(sorted_importance_power, 0)])

        #* calculate average and update
        sorted_weighted_importance_value = sorted_importance_value*sorted_importance_power
        sorted_importance_cumsum = torch.concatenate([float_head, \
                            torch.cumsum(sorted_weighted_importance_value, 0)])
        groupby_sum = sorted_importance_cumsum[leading_cell[1:]] - sorted_importance_cumsum[leading_cell[:-1]]
        groupby_powersum = sorted_power_cumsum[leading_cell[1:]] - sorted_power_cumsum[leading_cell[:-1]]
        # torch.cumsum has a bug making the result decrease even if all elements of the original array is positive. most possibly an underflow.
        groupby_sum = torch.where(groupby_sum < 0, 0, groupby_sum)
        groupby_powersum = torch.where(groupby_powersum < 0, 0, groupby_powersum) + 1e-10
        groupby_average = groupby_sum / groupby_powersum
        reordered_inter_level_importance_value = groupby_average

        #! calculate remove_empty_importance new value.
        #* calculate covariance between extend cell index and weights. This is to encorage the cells to slice off empty areas.
        sorted_cellid_wei_prod_cumsum = torch.concatenate([float_head, \
                                         torch.cumsum(ingredients["extend_cell_index"]*ingredients["raw_transmit_ratio"], 0)])
        sorted_extend_cell_index_cumsum = torch.concatenate([float_head, torch.cumsum(ingredients["extend_cell_index"], 0).float()])
        sorted_raw_transmit_ratio_cumsum = torch.concatenate([float_head, torch.cumsum(ingredients["raw_transmit_ratio"], 0)])

        sorted_cellid_wei_prod_sum = sorted_cellid_wei_prod_cumsum[leading_cell[1:]] - \
                                      sorted_cellid_wei_prod_cumsum[leading_cell[:-1]]
        sorted_extend_cell_index_sum = sorted_extend_cell_index_cumsum[leading_cell[1:]] - \
                                        sorted_extend_cell_index_cumsum[leading_cell[:-1]]
        sorted_raw_transmit_ratio_sum = sorted_raw_transmit_ratio_cumsum[leading_cell[1:]] - \
                                    sorted_raw_transmit_ratio_cumsum[leading_cell[:-1]]
        sorted_cell_count = leading_cell[1:] - leading_cell[:-1]
        cellid_n_wei_cov = sorted_cellid_wei_prod_sum / sorted_cell_count - \
                            sorted_extend_cell_index_sum*sorted_raw_transmit_ratio_sum / sorted_cell_count**2
        reordered_remove_empty_importance_value = cellid_n_wei_cov*cellid_n_wei_cov

        return reordered_cell_index, reordered_inter_level_importance_value, groupby_powersum, \
                                    reordered_remove_empty_importance_value, sorted_cell_count

    @torch.inference_mode()
    def update_importance(self, related_cell_index, importance_value, importance_power, update_inter_level_importance=True, update_rate=0.999):
        '''
        a bunch of cells at `related_cell_index` are affected by a training process, and they need to update their importance values. An affected cell is a leaf cell, the importance_value stands for the difference between the cell itself and the subcells created by `sample_extend`. The value will be applied on the cut edge of the cell(self.child_cut).

        #? the related_cell_index may contain repeatative cells, or cells that are not sampled(with self.child_cut[?] = -1). For the 1st, reduce them to one value; for the 2nd, discard them.

        related_cell_index: Tensor[cell_num]
        importance_value: Tensor[cell_num]
        '''
        #* normalize importance_value
        update_importance = self.inter_level_importance if update_inter_level_importance else self.remove_empty_importance

        #* find which edge_incell need to be updated
        # optional: check whether cells are sampled for extension (if not, `importance_value` will usually be 0 for them)
        related_child_cut = self.child_cut[related_cell_index, :]
        related_edge_idincell = self.edge_idincell_from_point(related_child_cut)
        # groupby_edge_valid = (groupby_edge_idincell != -1)

        cellwise_update_rate = torch.pow(update_rate, importance_power)
        cellwise_importance_last = update_importance[related_cell_index, related_edge_idincell]
        cellwise_importance_new = exponential_update(cellwise_importance_last, importance_value, cellwise_update_rate)

        #* update the importance
        if update_inter_level_importance:
            self.inter_level_importance[related_cell_index, related_edge_idincell] = cellwise_importance_new
        else:
            self.remove_empty_importance[related_cell_index, related_edge_idincell] = cellwise_importance_new
        
        if DEBUG and self.step % 100 == 0:
            with open(DEBUG_file, "a") as write_file:
                sample_id = self.step // 100 * 37831 % related_cell_index.shape[0]
                tag = " [inter_level_importance]: " if update_inter_level_importance else " [remove_empty_importance]: "
                update_str = tag + "cell " + str(related_cell_index[sample_id].item()) + \
                            " is updating its edge " + str(related_edge_idincell[sample_id].item()) + \
                            " with new value " + str(cellwise_importance_new[sample_id].item()) + \
                            " which without averaging is " + str(importance_value[sample_id].item()) + \
                            " and cellwise_update_rate " + str(cellwise_update_rate[sample_id].item()) + \
                            " and cellwise_importance_power " + str(importance_power[sample_id].item()) + \
                            " and momentum " + str(self.importance_momentum[0].item()) + "\n"
                
                write_file.write(update_str)

        importance_nan = torch.isnan(cellwise_importance_new)
        if torch.any(importance_nan):
            importance_nan_id = torch.nonzero(importance_nan)
            importance_nan_cell_index = related_cell_index[importance_nan_id.view(-1)[0]]
            nan_importance_value = importance_value[importance_nan_id.view(-1)[0]]
            nan_importance_power = importance_power[importance_nan_id.view(-1)[0]]
            nan_cellwise_update_rate = cellwise_update_rate[importance_nan_id.view(-1)[0]]
            with open(DEBUG_file, "a") as write_file:
                tag = " [inter_level_importance]: " if update_inter_level_importance else " [remove_empty_importance]: "
                update_str = tag + "updating nan! cell " + str(importance_nan_cell_index.item()) + \
                            " with new value nan! actually, " + str(importance_nan_id.shape[0]) + \
                            " different cells are getting nan! it happens with importance_value " + str(nan_importance_value.item()) +\
                            " and importance_power " + str(nan_importance_power.item()) + \
                            " and cellwise_update_rate " + str(nan_cellwise_update_rate.item()) + \
                            ". Note that momentum is " + str(self.importance_momentum[0].item()) + "\n"
                write_file.write("================== UPDATE IMPORTANCE =================\n")
                write_file.write(update_str)
                write_file.write("======================================================\n")
            assert 0

    # ! detailed function definition starts here
    def organize_layer_0(self):
        self.point_index[:1,:] = torch.arange(4, device=self.point_index.device).view(1,4)
        self.edge_index[:1] = torch.arange(6, device=self.edge_index.device).view(1,6)
        self.layer[:1] = 0

        point_xyz = self.get_tetra_xyz(self.aabb)
        self.xyz[:4,:] = point_xyz
        with torch.no_grad():
            self.field[:4, :].uniform_(0,0.1)
            self.field[0, :].view(2,2,8)[:,0,:] += 1
            self.field[1, :].view(4,2,4)[:,0,:] += 1
            self.field[2, :].view(8,2,2)[:,0,:] += 1
            self.field[3, :].view(16,2,1)[:,0,:] += 1

        self.cell_offset[0] = 1
        self.edge_offset[0] = 6
        self.point_offset[0] = 4
    
    def search_layer_0(self, xyz):
        sample_size = xyz.shape[0]

        cell_id = torch.zeros(sample_size, dtype=torch.int64, device=xyz.device)
        cell_xyz = self.xyz[:4, :].view(1,4,3).expand(sample_size, 4, 3)

        cut = self.child_cut[:1, :].view(1, 2).expand(sample_size, 2)
        activation_layer = self.activation_layer[:1].expand(sample_size)

        return {"cell_id":          cell_id, 
                "xyz":              cell_xyz, 
                "cut":              cut, 
                "activation_layer": activation_layer}

    def search_layer_i(self, xyz, parent_cells, i):
        sample_num = xyz.shape[0]

        parent_id = parent_cells["cell_id"]
        parent_xyz = parent_cells["xyz"]
        parent_cut = parent_cells["cut"]
        parent_activation_layer = torch.min(parent_cells["activation_layer"], self.max_layer_num)

        children_choice = divide_into_children(xyz, parent_cells)
        # **child_cell_id
        child_cell_id = self.child_index[parent_id, children_choice]
        # **child_valid, mid variable
        child_valid = torch.logical_and(parent_id != -1, child_cell_id != -1)
        child_valid = torch.logical_and(child_valid, parent_activation_layer == i)
        #? maybe later use the dropout line below to keep stable
        # child_valid = torch.nn.Dropout(p=0.05, inplace=True)(child_valid)
        # **child_vertex_id, mid variable, now represented as abandoned_vertex & child_vertex_id_substitute
        # only one vertex of the parent cell will change when divided into a subcell
        ## element of abandoned_vertex is in 0,1,2,3
        abandoned_vertex = torch.gather(parent_cut, 1, 1-children_choice.view(-1,1)).view(-1)

        child_vertices_id_substitute = self.point_index[child_cell_id, abandoned_vertex]
        # **child_xyz
        child_xyz = parent_xyz.clone()
        cut_point_xyz = parent_xyz[torch.arange(sample_num).view(-1,1), parent_cut, :]
        child_xyz_substitute = torch.sum(cut_point_xyz, axis=1) / 2
        child_xyz[torch.arange(sample_num), abandoned_vertex, :] = child_xyz_substitute
        # **child_cut
        child_cut = self.child_cut[child_cell_id, :]
        # **child_activation_layer
        child_activation_layer = self.activation_layer[child_cell_id]

        # filter out inactivated cells
        child_cell_id          = torch.where(child_valid,              child_cell_id,          parent_id)
        child_xyz              = torch.where(child_valid.view(-1,1,1), child_xyz,              parent_xyz)
        child_cut              = torch.where(child_valid.view(-1,1),   child_cut,              parent_cut)
        child_activation_layer = torch.where(child_valid,              child_activation_layer, parent_activation_layer)

        return_dict = {"cell_id":          child_cell_id, 
                       "xyz":              child_xyz, 
                       "cut":              child_cut, 
                       "activation_layer": child_activation_layer}
        return return_dict

    def search_extend(self, xyz, cells):
        cells = self.search_layer_i(xyz, cells, self.max_layer_num)
        return cells

    @torch.inference_mode()
    def remove_last_sampled(self, initializing):
        child_start = self.cell_offset[0]
        child_end = self.cell_offset[0] + self.extend_cell_offset[0]

        point_start = self.point_offset[0]
        point_end = self.point_offset[0] + self.extend_point_offset[0]

        #** clean up parent connection
        parent_cell_index = self.parent_index[child_start:child_end:2]            
        
        self.child_index[parent_cell_index, :] = -1
        self.child_cut[parent_cell_index, :] = -1
        self.activation_layer[parent_cell_index] = -1

        #** clean up child values
        self.parent_index[child_start:child_end] = -1
        self.point_index[child_start:child_end, :] = -1
        self.child_index[child_start:child_end, :] = -1
        self.child_cut[child_start:child_end, :] = -1
        self.activation_layer[child_start:child_end] = -1
        self.layer[child_start:child_end] = -1

        #** clean up point info
        self.field[point_start:point_end, :] = 0
        self.xyz[point_start:point_end, :] = 0
        self.parent_points[point_start:point_end, :] = -1

        #** clean optimizer state
        if not initializing:
            utils.reset_global_optimizers_patial(self.cell_offset[0])

        #** clean up overall extend labels
        self.extend_cell_offset[0] = 0
        self.extend_point_offset[0] = 0

    @torch.inference_mode()
    def merge_extend(self, merged_cell_count):
        is_leaf_cell = self.is_leaf_cell()
        #? use a more elegant function for importance synchronization, rather than "max"
        leaf_edge_importance = self.importance*is_leaf_cell.view(-1,1)
        valid_edge_importance = self.synchronize_cell2edge(leaf_edge_importance, "max")
        valid_edge_importance = torch.where(torch.isnan(valid_edge_importance), -1, valid_edge_importance) \
                                    * is_leaf_cell.view(-1,1)
        #? number of chosen edges is fixed to be merged_cell_count, but each cell may at most include 6 chosen edges, so the number of actual chosen cell falls far below merged_cell_count.
        sorted_valid_edge_importance, edge_candidate_index_1d = torch.sort(valid_edge_importance.view(-1), descending=True)
        edge_candidate_index_1d = edge_candidate_index_1d[:merged_cell_count]
        edge_candidate = torch.zeros(self.max_cell_count, 6, dtype=torch.bool, device=self.activation_layer.device)
        edge_candidate.view(-1).scatter_(0, edge_candidate_index_1d, torch.ones_like(edge_candidate_index_1d, dtype=torch.bool, device=self.activation_layer.device))
        cell_candidate = (torch.sum(edge_candidate, axis=1) != 0)
        # the following line is only to remove the edge candidate with the least importance, which may create more subcells than expected
        edge_candidate = torch.logical_or(edge_candidate, ~is_leaf_cell.view(-1,1))
        edge_candidate = self.synchronize_cell2edge(edge_candidate, "and")
        edge_candidate = torch.logical_and(edge_candidate, is_leaf_cell.view(-1,1))
        edge_candidate = self.synchronize_cell2edge(edge_candidate, "or")
        edge_determined = torch.zeros_like(edge_candidate)
        iteration = 0
        while True:
            #** sample iteration by iteration
            stride = merged_cell_count // 25 + 1
            window_size = merged_cell_count // 5 + 1
            bound_start = torch.min(stride*iteration, \
                        torch.tensor([6*self.max_cell_count-window_size], dtype=stride.dtype, device=stride.device))
            average_importance = torch.mean(sorted_valid_edge_importance[bound_start : bound_start+window_size])
            iteration += 1
            # bound of sample_value increase with importance
            sample_value = self.importance - average_importance
            # sample rate decrease with iteration count
            sample_rate = torch.rand(self.max_cell_count,6, device=self.activation_layer.device) * torch.max(sample_value) * 10/(iteration+10)

            #** sample a few bunch of edges, at most one edge per cell
            chosen_edge_stage1 = torch.logical_or(edge_determined, \
                            torch.logical_and(sample_value >= sample_rate, edge_candidate))
            _, chosen_edge_stage2 = self.choose_edge(chosen_edge_stage1, \
                                                torch.sum(chosen_edge_stage1, axis=1), \
                                                torch.any(chosen_edge_stage1, axis=1))
            chosen_edge_stage2 = self.synchronize_cell2edge(chosen_edge_stage2, "or")

            #** make sure no conflict exists
            edge_valid_temp = self.broadcast_n_remove_conflict(chosen_edge_stage2, is_leaf_cell)
            edge_candidate = torch.logical_and(edge_valid_temp, edge_candidate)

            candidate_edge_num = torch.sum(edge_candidate, axis=1) * is_leaf_cell

            edge_determined = torch.where(candidate_edge_num.view(-1,1).expand(-1,6) == 1, edge_candidate, edge_determined)

            if torch.all(candidate_edge_num <= 1):
                break
        # note, return of broadcast_n_remove_conflict only determines the `invalid`s, so only cells with a fixed chosen edge should be taken into account
        chosen_cell = (torch.sum(edge_candidate, axis=1) == 1)
        edge_valid_for_original = edge_candidate
        #? .int() is for argmax_cpu, and might be removed after debugging
        edge_valid_for_original_idincell = torch.argmax(edge_valid_for_original.int(), axis=1)

        #? this is to seperate old cells from newly merged cells, maybe move it to a better position later.
        self.cell_offset[1] = self.cell_offset[0]
        self.point_offset[1] = self.point_offset[0]
        self.apply_sampled(edge_valid_for_original)
        print("an additional ", self.extend_cell_offset[0], " cells will be added.")
        if DEBUG:
            with open(DEBUG_file, "a") as write_file:
                write_file.write("an additional "+ str(self.extend_cell_offset[0].item()) + " cells will be added.\n")

        #! start setting structural variables
        child_start = self.cell_offset[0]
        child_end = self.cell_offset[0] + self.extend_cell_offset[0]
        parent_cell_index = self.parent_index[child_start:child_end]
        parent_child_cut = self.child_cut[parent_cell_index, :]
        parent_neighbor = self.neighbor[:, parent_cell_index, :]

        child_index_relative = torch.arange(child_end - child_start, dtype=torch.int64, device=self.cell_offset.device)
        child_01 = child_index_relative % 2
        child_index = child_index_relative + child_start
        abandoned_vertex = torch.gather(parent_child_cut, 1, (1-child_01).view(-1,1))
        maintained_vertex = torch.gather(parent_child_cut, 1, child_01.view(-1,1))
        
        #** set the values of activation_layer & layer
        # this next line might be risky, if child cells do not come in pairs, but it is safe for now
        edge_layer_incell = self.layer.view(-1,1).expand(-1,6).clone()
        edge_layer = self.synchronize_cell2edge(edge_layer_incell, "max")
        cell_valid_for_original = torch.logical_and(torch.any(edge_valid_for_original, axis=1), is_leaf_cell)
        cell_activation_layer = torch.sum(edge_layer*edge_valid_for_original, axis=1).int()
        self.activation_layer = torch.where(cell_valid_for_original, cell_activation_layer+1, self.activation_layer)
        self.layer[child_start:child_end] = self.activation_layer[parent_cell_index]

        #** set new edge indices
        # check how many new cells each new subdivision(each new point) cuts, only include leaf edges
        child_edge_index = self.edge_index[parent_cell_index].clone()

        selected_edge_idincell = edge_valid_for_original_idincell[parent_cell_index]

        new_edge_offset, unselected_point_idincell, selected_point_idincell, \
                        new_edge_cellindex_unselected, new_edge_cellindex_selected, full_subedge_count = \
                    self.arange_new_edge_for_parent_cell(parent_cell_index, selected_edge_idincell)
        
        unselected_edge_point_idincell = torch.stack((abandoned_vertex.expand(-1,2), unselected_point_idincell), axis=-1)
        unselected_edge_idincell = self.edge_idincell_from_point(unselected_edge_point_idincell)
        unselected_edge_index = (new_edge_offset + new_edge_cellindex_unselected + self.edge_offset[0])
        child_edge_index.scatter_(1, unselected_edge_idincell, unselected_edge_index)

        selected_edge_index_full = new_edge_offset + new_edge_cellindex_selected + self.edge_offset[0]
        selected_edge_index = torch.gather(selected_edge_index_full, 1, child_01.view(-1,1))
        child_edge_index.scatter_(1, selected_edge_idincell.view(-1,1), selected_edge_index)

        self.edge_offset[0] += full_subedge_count
        self.edge_index[child_start:child_end, :] = child_edge_index

        self.point_offset[0] += self.extend_point_offset[0]
        self.cell_offset[0] += self.extend_cell_offset[0]
        self.extend_point_offset[0] = 0
        self.extend_cell_offset[0] = 0

        self.edge_sorted[0] = False

    @torch.inference_mode()
    def sample_extend(self):
        #? sampling for all cells may be costly, maybe sample for leaf cells only
        cell_num = self.max_cell_count.to(self.activation_layer.device)
        is_leaf_cell = self.is_leaf_cell()
        # the following variables are only valid for condition `is_leaf_cell`, that is, only leaf of the tree can choose edges.
        edge_valid = torch.ones(self.max_cell_count, 6, device=self.activation_layer.device)
        cell_edge_num = 6*torch.ones_like(self.activation_layer, device=self.activation_layer.device)
        i = torch.tensor(0, dtype=torch.int32, device=self.activation_layer.device)
        while True:
            sample_rate = 1/(20-i)
            if i < 19:
                i += 1
            # **Choose some cells, choose one edge for each of them. then, if an edge is at the same time labeled `chosen` and `not chosen`, choose it.
            sample_value = (cell_edge_num-1)/5 * torch.rand(cell_num, device=edge_valid.device)
            cell_chosen_coarse = torch.logical_and(torch.logical_and(sample_value < sample_rate, \
                                                                        cell_edge_num != 0), \
                                                                        is_leaf_cell)
            _, edge_chosen = self.choose_edge(edge_valid, cell_edge_num, cell_chosen_coarse)
            edge_chosen = self.synchronize_cell2edge(edge_chosen, "or")
            edge_valid = self.broadcast_n_remove_conflict(edge_chosen, is_leaf_cell)

            cell_edge_num = torch.sum(edge_valid, axis=1)
            if torch.all(torch.logical_or(cell_edge_num <= 1, ~is_leaf_cell)):
                break
    
        return edge_valid

    @torch.inference_mode()
    def apply_sampled(self, edge_valid):
        extend_offset = self.cell_offset[0]
        is_leaf_cell = self.is_leaf_cell().view(-1,1).expand(-1,6)
        edge_valid_leaf = torch.logical_and(is_leaf_cell, edge_valid)
        # **each edge which is valid for subdivision corresponds to a point in the extend layer, we need to get each edge an index(the same edges in different cells correspond to the same point index).
        # set each unique edge a number and remove the conflicts
        trace_edge = torch.ones(6*self.max_cell_count, device=edge_valid.device).view(self.max_cell_count, 6)
        trace_edge = self.synchronize_cell2edge(trace_edge, "unique")
        # each valid leaf cell contains an edge for subdivision, convert the edge number to the point index causing the subdivision
        trace_edge_in_cell = torch.sum(edge_valid_leaf*trace_edge, axis=1)
        sorted_trace_edge_in_cell, sort_index = torch.sort(trace_edge_in_cell, descending=True)
        step_posi = sorted_trace_edge_in_cell[:-1] - sorted_trace_edge_in_cell[1:]
        head = torch.tensor([0], dtype=torch.int32, device=step_posi.device)
        sorted_trace_edge_in_cell_normalized = torch.concatenate((head, torch.cumsum(step_posi != 0, dim=0)))
        # get "which cell index is divided by which point index & which point can be represented by which cell"
        parent_cell_num = torch.sum(edge_valid_leaf)
        parent_cell_index = sort_index[:parent_cell_num]
        subdivision_point_index = sorted_trace_edge_in_cell_normalized[:parent_cell_num] + self.point_offset[0]

        one_cell_per_point_mid_index = torch.nonzero(step_posi).view(-1)
        one_cell_per_point = sort_index[one_cell_per_point_mid_index]

        # **connect parent and children
        parent_child_index = torch.arange(2*parent_cell_num, device=edge_valid.device).view(-1,2) + self.cell_offset[0]

        #? .int() is for argmax_cpu, and might be removed after debugging
        parent_edge_index = torch.argmax(edge_valid[parent_cell_index, :].int(), axis=1)
        parent_child_cut = self.selected_point_idincell_from_edge(parent_edge_index)

        self.child_index[parent_cell_index, :] = parent_child_index
        self.child_cut[parent_cell_index, :] = parent_child_cut
        self.activation_layer[parent_cell_index] = self.max_layer_num

        # **construct children cells
        children_parent_index = parent_cell_index.view(-1,1).expand(-1,2)

        abandoned_vertex = torch.stack((parent_child_cut[:,1], parent_child_cut[:,0]), axis=1).view(-1,2,1)
        parent_point_index = self.point_index[parent_cell_index, :]
        children_point_index = parent_point_index.reshape(-1,1,4).expand(-1,2,4) \
                            .scatter(2, abandoned_vertex.long(), subdivision_point_index.view(-1,1,1).expand(-1,2,1))

        child_start = self.cell_offset[0]
        child_end = self.cell_offset[0] + 2*parent_cell_num
        self.parent_index[child_start:child_end] = children_parent_index.reshape(-1)
        self.point_index[child_start:child_end, :] = children_point_index.view(-1,4)
        self.child_index[child_start:child_end, :] = -1
        self.child_cut[child_start:child_end, :] = -1
        self.activation_layer[child_start:child_end] = -1
        self.layer[child_start:child_end] = self.max_layer_num

        # **construct children points
        divided_point_idincell = self.child_cut[one_cell_per_point, :].long()
        parent_point_index_one_cell_per_point = self.point_index[one_cell_per_point, :]
        divided_point_index = torch.gather(parent_point_index_one_cell_per_point, 1, divided_point_idincell)
        child_xyz = torch.sum(self.xyz[divided_point_index, :], axis=1) / 2

        point_start = self.point_offset[0]
        point_end = self.point_offset[0] + one_cell_per_point.shape[0]
        self.field[point_start:point_end, :] = 0
        self.xyz[point_start:point_end, :] = child_xyz
        self.parent_points[point_start:point_end, :] = divided_point_index

        # **change overall extend label
        self.extend_cell_offset[0] = child_end - child_start
        self.extend_point_offset[0] = point_end - point_start

    @torch.inference_mode()
    def apply_newest_layer_field(self):
        newest_layer_point_start = self.point_offset[1]
        newest_layer_point_end = self.point_offset[0]
        if self.point_offset[1] == 0:
            return

        newest_layer_point_parent_point = self.parent_points[newest_layer_point_start:newest_layer_point_end, :]
        newest_layer_point_field_base_value = torch.sum(self.field[newest_layer_point_parent_point, :], axis=1) / 2
        newest_layer_point_field_change = self.field[newest_layer_point_start:newest_layer_point_end, :]
        self.field[newest_layer_point_start:newest_layer_point_end, :] = newest_layer_point_field_base_value + \
                                                                        newest_layer_point_field_change

    # ! second level detailed function definition starts here
    def get_tetra_xyz(self, aabb):
        #TODO: get the tetrahedron cover of `aabb`, only xyz values, so shape (4,3)
        # u = 0.25
        # t = 0.75
        # return torch.tensor([[u,u,u], [u,u,t], [u,t,u], [t,u,u]], dtype=torch.float32, device=aabb.device)
        return torch.tensor([[-0.25,0.25,1.5], [-0.25,0.25,-1.5], [2.75,0.25,0], [0.5,3.25,0]])
        # return torch.tensor([[0.25,0.25,0.25], [0.25,0.75,0.25], [0.75,0.5,0.25], [0.5,0.5,0.75]])

    def is_leaf_cell(self):
        is_leaf_candidate = torch.logical_or(self.activation_layer>=self.max_layer_num, self.activation_layer==-1)
        is_original_cell = (torch.arange(self.max_cell_count, device=self.activation_layer.device) < self.cell_offset[0])
        not_reach_max_layer = (self.layer < self.max_layer_num-1)
        is_leaf_cell = torch.logical_and(torch.logical_and(is_leaf_candidate, is_original_cell), not_reach_max_layer)
        return is_leaf_cell

    def choose_edge(self, edge_valid, cell_edge_num, cell_chosen):
        cell_num = edge_valid.shape[0]
        edge_chosen_index_in_valid = ( torch.rand(cell_num, device=self.activation_layer.device) * cell_edge_num ).int() + 1
        confirmed = torch.zeros(cell_num, dtype=torch.bool, device=edge_valid.device)
        edge_chosen_index = torch.zeros(cell_num, dtype=torch.int64, device=edge_valid.device)
        edge_chosen = torch.zeros(cell_num, 6, dtype=torch.bool, device=edge_valid.device)
        valid_edge_count_prefix = torch.cumsum(edge_valid, dim=1)
        for i in range(6):
            chosen_condition = torch.logical_and(valid_edge_count_prefix[:,i]==edge_chosen_index_in_valid, ~confirmed)
            confirmed = torch.logical_or(chosen_condition, confirmed)
            edge_chosen_index = torch.where(chosen_condition, i, edge_chosen_index)

        edge_chosen = torch.scatter(edge_chosen, 1, edge_chosen_index.view(-1,1), cell_chosen.view(-1,1))
        return edge_chosen_index, edge_chosen

    def broadcast_n_remove_conflict(self, edge_chosen, is_leaf_cell):
        # **conflictions may happen in the following scenario: a and b choose two different common edges with c. To fix this, choose only one edge in each conflicting cell and invalidate others. if an edge is at the same time labeled `invalid` and `not invalid`, invalidate it.
        # note, cells with only one valid edge never get invalidated, because all conflicting edges have been invalidated by the next step in the last round, and will not be chosen this round
        cell_chosen_edge_num = torch.sum(edge_chosen, axis=1)
        cell_related = torch.logical_and(cell_chosen_edge_num != 0, is_leaf_cell)
        edge_chosen_index, edge_chosen = self.choose_edge(edge_chosen, cell_chosen_edge_num, cell_related)
        edge_chosen = torch.logical_or(edge_chosen, ~is_leaf_cell.view(-1,1))
        edge_chosen = self.synchronize_cell2edge(edge_chosen, "and")
        edge_chosen = torch.logical_and(edge_chosen, is_leaf_cell.view(-1,1))
                
        cell_chosen = (torch.sum(edge_chosen, axis=1) == 1)
        # **In each cell, if an edge is chosen, all other edges are invalidated. if an edge is at the same time labeled `invalid` and `not invalid`, invalidate it.
        edge_valid = (~cell_chosen).view(-1,1).expand(-1,6).clone()
        edge_valid.scatter_(1, edge_chosen_index.view(-1,1), torch.ones_like(cell_chosen).view(-1,1))
        edge_valid = self.synchronize_cell2edge(edge_valid, "and")

        return edge_valid

    def synchronize_cell2edge(self, sync_src, reduction_func):
        '''
        synchronize the value of each edge,  because they(`sync_src`) are stored and updated at the structure of cells. do reduction along the tree to a partial root, and then spread the reduction results back to the leaves
        '''
        # most reduction_func can be implemented as a special case of "sum", except "max". we implement it as below.
        if reduction_func == "max":
            buffer = torch.zeros_like(sync_src)
            
            max_cell2sortededge_map_inner = torch.argsort(sync_src.view(-1), descending=True)
            sorted_edge_index_inner = self.edge_index.view(-1)[max_cell2sortededge_map_inner]
            sorted_edge_index, max_cell2sortededge_map_outter = torch.sort(sorted_edge_index_inner, stable=True, descending=True)
            max_cell2sortededge_map = max_cell2sortededge_map_inner[max_cell2sortededge_map_outter]
            sync_src_indexsort = sync_src.view(-1)[max_cell2sortededge_map]

            sorted_edge_changing = (sorted_edge_index[1:]-sorted_edge_index[:-1] != 0)
            head = torch.tensor([0], dtype=sorted_edge_index.dtype, device=sorted_edge_index.device)
            leading_edge = torch.concatenate((head, 1+torch.nonzero(sorted_edge_changing).view(-1)), axis=0)
            leadedby_edge = torch.concatenate((head, torch.cumsum(sorted_edge_changing, dim=0)), axis=0)
            sorted_max_edge = leading_edge[leadedby_edge]
            sorted_max_sync_src = sync_src_indexsort[sorted_max_edge]

            buffer.view(-1).scatter_(0, max_cell2sortededge_map, sorted_max_sync_src)
            return buffer

        #? the following line might be deleted when integrated
        if reduction_func != "average" and reduction_func != "sum":
            sync_src = sync_src.int()
        self.sort_for_edge()
        sync_src_indexsort = sync_src.view(-1)[self.cell2sortededge_map]
        head = torch.zeros([1], dtype=sync_src_indexsort.dtype, device=sync_src_indexsort.device)
        cumsum = torch.concatenate((head, torch.cumsum(sync_src_indexsort, dim=0)), axis=0)
        cumsum_prefix = cumsum[self.sorted_edge_offset]
        groupby_sum = cumsum_prefix[1:] - cumsum_prefix[:-1]
        # reduction_func is a string in ["and", "or", "sum", "average", "unique", ...]
        groupby_count = self.sorted_edge_offset[1:] - self.sorted_edge_offset[:-1]
        if reduction_func =="sum":
            groupby_result = groupby_sum
        elif reduction_func == "and":
            groupby_result = (groupby_sum == groupby_count)
        elif reduction_func == "or":
            groupby_result = ~(groupby_sum == 0)
        elif reduction_func == "average":
            groupby_result = groupby_sum / groupby_count
        elif reduction_func == "unique":
            #? a waste to generate groupby_sum
            groupby_result = torch.arange(groupby_sum.shape[0], device=groupby_sum.device) + 1
        elif reduction_func == "broadcast":
            groupby_result = sync_src_indexsort[self.sorted_edge_offset[:-1]]
        else:
            import sys
            print("reduction_func ", reduction_func, " not supported.")
            sys.exit(1)

        edge_buffer = torch.zeros_like(self.edge_importance, dtype=groupby_result.dtype)
        edge_buffer.scatter_(0, self.sorted_edge_index_unique, groupby_result)
        #? this might get some strange values from edge_buffer[-1]
        return edge_buffer[self.edge_index.view(-1)].view(sync_src.shape)

    def arange_new_edge_for_parent_cell(self, parent_cell_index, selected_edge_idincell):
        '''
        #? 2*parent_cell_count=child_cell_count or parent_cell_count?
        selected_edge_idincell: Tensor[parent_cell_count] # in range(0,6)
        # for example: edge_i covers the range from edge_start to edge_end
        ret_offset: Tensor[parent_cell_count, 6] # equal to edge_start
        ret_cellindex_unselected: Tensor[parent_cell_count, 2] # range from 2 to edge_end-edge_start+2
        ret_cellindex_selected: Tensor[parent_cell_count, 2] # range from 0 to 1
        '''
        selected_point_idincell = self.selected_point_idincell_from_edge(selected_edge_idincell)
        unselected_point_idincell = self.unselected_point_idincell_from_edge(selected_edge_idincell)

        selected_point_index = self.point_index[parent_cell_index.view(-1,1), selected_point_idincell]
        unselected_point_index = self.point_index[parent_cell_index.view(-1,1), unselected_point_idincell]
        selected_edge_index = self.edge_index[parent_cell_index, selected_edge_idincell]

        # do sorting acording to (main order) edge_index and (secondary order) unselected_point_index
        inner_sorting_indices = torch.argsort(unselected_point_index.view(-1))
        selected_edge_index_expand = selected_edge_index.view(-1,1).expand(-1,2).clone()
        selected_edge_index_expand_innersort = selected_edge_index_expand.view(-1)[inner_sorting_indices]
        outer_sorting_indices = torch.argsort(selected_edge_index_expand_innersort, stable=True)

        twolevel_sorting_indices = inner_sorting_indices[outer_sorting_indices]
        twolevel_sorted_edge = selected_edge_index_expand.view(-1)[twolevel_sorting_indices]
        twolevel_sorted_unselected_point = unselected_point_index.view(-1)[twolevel_sorting_indices]

        # set offset and cellindex for new subedges cutting one face of the parent cell
        twolevel_sorted_edge_changing = (twolevel_sorted_edge[1:]-twolevel_sorted_edge[:-1] != 0)
        twolevel_sorted_point_changing = (twolevel_sorted_unselected_point[1:]-twolevel_sorted_unselected_point[:-1] != 0)
        twolevel_sorted_changing = torch.logical_or(twolevel_sorted_edge_changing,
                                                    twolevel_sorted_point_changing)

        reindex_nohead = torch.cumsum(twolevel_sorted_changing, dim=0)
        head = torch.tensor([0], dtype=reindex_nohead.dtype, device=reindex_nohead.device)
        reindex = torch.concatenate((head, reindex_nohead), axis=0)

        leading_subedge = torch.concatenate((head, 1+torch.nonzero(twolevel_sorted_edge_changing).view(-1)), axis=0)
        leadedby_subedge = torch.concatenate((head, torch.cumsum(twolevel_sorted_edge_changing, dim=0)), axis=0)
        leadedby_subedge_sorted_indices = leading_subedge[leadedby_subedge]
        twolevel_sorted_offset = reindex[leadedby_subedge_sorted_indices]
        twolevel_sorted_cellindex = reindex - twolevel_sorted_offset

        ## note, each divided edge in the parent cell creates 2+N new edges, where N is the number of adjacent points of the edge. thus the i+1'th edge should be offset 2*i+\sigma{N}
        twolevel_sorted_offset_plus2 = twolevel_sorted_offset + 2*leadedby_subedge
        offset_biased = torch.scatter(unselected_point_index.view(-1), 0, \
                                    twolevel_sorting_indices, twolevel_sorted_offset_plus2).view(-1,2)
        cellindex_unselected_biased = torch.scatter(unselected_point_index.view(-1), 0, \
                                    twolevel_sorting_indices, twolevel_sorted_cellindex).view(-1,2)
        ret_offset = offset_biased
        ret_cellindex_unselected = cellindex_unselected_biased + 2

        # set offset and cellindex for new subedges divided from one edge of the parent cell
        twolevel_sorted_selected_point = selected_point_index.view(-1)[twolevel_sorting_indices]
        leading_subedge_point_index = twolevel_sorted_selected_point[leadedby_subedge_sorted_indices]
        twolevel_sorted_issubedge = (leading_subedge_point_index != twolevel_sorted_selected_point)
        ret_cellindex_selected = torch.scatter(selected_point_index.view(-1), 0, \
                                    twolevel_sorting_indices, twolevel_sorted_issubedge.long()).view(-1,2)

        full_subedge_count = 1+torch.sum(twolevel_sorted_changing)
        full_subedge_count += 2*leadedby_subedge[-1] + 2

        # ret_cellindex_unselected is for subedge: division point A & unselected_point_idincell
        # ret_cellindex_selected is for subedge: division point A & selected_point_idincell
        return ret_offset, unselected_point_idincell, selected_point_idincell, \
                           ret_cellindex_unselected,  ret_cellindex_selected, full_subedge_count

    # ! third level detailed function definition starts here
    def selected_point_idincell_from_edge(self, selected_edge_idincell):
        '''
        selected_edge_idincell: Tensor[parent_cell_count] # in range(0,6)
        ret_point_index: Tensor[parent_cell_count, 2] # if cell not selected, use -1; else in range(0,4)
        ret_point_index follows 0,1,2,3 order
        '''
        edge2selected_point = torch.tensor([[0,1],[0,2],[0,3],[1,2],[1,3],[2,3]], dtype=torch.int32, device=selected_edge_idincell.device)
        ret_point_idincell = edge2selected_point[selected_edge_idincell, :]
        return ret_point_idincell

    def unselected_point_idincell_from_edge(self, selected_edge_idincell):
        '''
        selected_edge_idincell: Tensor[parent_cell_count] # in range(0,6)
        ret_point_index: Tensor[parent_cell_count, 2] # if cell not selected, use -1; else in range(0,4)
        ret_point_index follows right-handed-system order
        '''
        edge2unselected_point = torch.tensor([[2,3],[3,1],[1,2],[3,0],[2,0],[0,1]], dtype=torch.int32, device=selected_edge_idincell.device)
        ret_point_idincell = edge2unselected_point[selected_edge_idincell, :]
        return ret_point_idincell

    def edge_idincell_from_point(self, point_idincell):
        '''
        point_idincell: Tensor[..., 2] # in range(0,4)
        ret_edge_idincell: Tensor[...] # in range(0,6)
        '''
        point2edge = torch.tensor([[-1,0,1,2], [0,-1,3,4], [1,3,-1,5], [2,4,5,-1]], dtype=torch.int64, device=point_idincell.device)

        point_idincell_flat = point_idincell.view(-1,2)
        edge_idincell_flat = point2edge[point_idincell_flat[:,0], point_idincell_flat[:,1]]
        ret_edge_idincell = edge_idincell_flat.view(point_idincell.shape[:-1])
        return ret_edge_idincell

    def sort_for_edge(self):
        #? still lack checking for invalid slots, that is, for `index >= self.max_cell_count`, we still have to manually set their `sync_to_trigger` values to a preset large constant value.
        #? memorize to change self.syncing_sort to False after subdivision
        if not self.edge_sorted or self.sorted_edge_offset == None:
            sorted_edge_index, self.cell2sortededge_map = torch.sort(self.edge_index.view(-1), descending=True)
            self.sorted_edge_index = sorted_edge_index
            offset = torch.nonzero(sorted_edge_index[1:] - sorted_edge_index[:-1]).view(-1)
            head = torch.zeros([1], dtype=offset.dtype, device=offset.device)
            # please note, `sorted_edge_index[offset[-1]:] == -1` necessarily happens, and the element number is not 0. that is why we do not take these elements into account in the following.
            self.sorted_edge_offset = torch.concatenate((head, offset+1), axis=0)
            self.sorted_edge_index_unique = sorted_edge_index[offset]
            self.edge_sorted[0] = True
        else:
            if self.sorted_edge_offset.device != self.activation_layer.device:
                self.sorted_edge_offset = self.sorted_edge_offset.to(self.activation_layer.device)
                self.sorted_edge_index_unique = self.sorted_edge_index_unique.to(self.activation_layer.device)

# pylint: disable=attribute-defined-outside-init
class FlexNerfField(Field):
    """Tetrahedra NeRF model

    Args:
        config: Basic NeRF configuration to instantiate model
    """

    def __init__(
        self,
        aabb,
        min_resolution: int = 4,
        feature_dim: int = 32,
        den_rgb_feat_dim: int = 32,
        max_layer_num: int = 40,
        max_point_per_layer: int = 1e5,
        num_layers: int = 3,
        hidden_dim: int = 128,
        geo_feat_dim: int = 15,
        num_layers_color: int = 3,
        hidden_dim_color: int = 64,
        spatial_distortion: Optional[SpatialDistortion] = None,
        implementation: Literal["tcnn", "torch"] = "tcnn",
    ) -> None:
        super().__init__()
        self.register_buffer("aabb", aabb)
        self.den_rgb_feat_dim = den_rgb_feat_dim
        self.feature_dim = feature_dim

        self.geo_feat_dim = geo_feat_dim
        self.spatial_distortion = spatial_distortion

        self.direction_encoding = SHEncoding(
            levels=4,
            implementation=implementation,
        )

        self.multi_layer_tetra = MultiLayerTetra(aabb, feature_dim, max_layer_num, 8*max_point_per_layer, max_point_per_layer)

        def split_sin_func(input_feature):
            input_feature_p1, input_feature_p2 = torch.split(input_feature, [16,16], -1)
            sin_feature = torch.cat([input_feature_p1, torch.sin(10*input_feature_p2)], dim=-1)
            return sin_feature
        self.split_sin_func = split_sin_func

        self.mlp_base_mlp = MLP(
            in_dim=den_rgb_feat_dim,
            num_layers=num_layers,
            layer_width=hidden_dim,
            out_dim=geo_feat_dim+1,
            activation=nn.LeakyReLU(),
            out_activation=None,
            implementation=implementation,
        )

        self.mlp_head = MLP(
            in_dim=self.direction_encoding.get_out_dim() + self.geo_feat_dim,
            num_layers=num_layers_color,
            layer_width=hidden_dim_color,
            out_dim=3,
            activation=nn.LeakyReLU(),
            out_activation=nn.Sigmoid(),
            implementation=implementation,
        )

    def get_density(self, ray_samples: RaySamples) -> Tuple[Tensor, Tensor]:
        """Computes and returns the densities."""
        if self.spatial_distortion is not None:
            positions = ray_samples.frustums.get_positions()
            positions = self.spatial_distortion(positions)
            positions = (positions + 2.0) / 4.0
        else:
            positions = SceneBox.get_normalized_positions(ray_samples.frustums.get_positions(), self.aabb)
        # Make sure the tcnn gets inputs between 0 and 1.
        selector = ((positions > 0.0) & (positions < 1.0)).all(dim=-1)
        positions = positions * selector[..., None]
        self._sample_locations = positions
        if not self._sample_locations.requires_grad:
            self._sample_locations.requires_grad = True
        positions_flat = positions.view(-1, 3)

        feature, extend_feature, old_feature, cell_id, extend_cell_id, importance = self.multi_layer_tetra(positions_flat, use_extend=True)

        def feature2out(input_feature):
            # sin_feature = self.split_sin_func(input_feature)
            sin_feature = input_feature
            density_feature = sin_feature[:, :self.den_rgb_feat_dim].contiguous()
            rgb_feature = sin_feature[:, -self.den_rgb_feat_dim:].contiguous()
            density_before_activation = self.mlp_base_mlp(density_feature).view(*ray_samples.frustums.shape, -1)[..., :1].contiguous()
            rgb_embedding = self.mlp_base_mlp(rgb_feature).view(*ray_samples.frustums.shape, -1)[..., 1:].contiguous()

            self._density_before_activation = density_before_activation

            # Rectifying the density with an exponential is much more stable than a ReLU or
            # softplus, because it enables high post-activation (float32) density outputs
            # from smaller internal (float16) parameters.
            density = trunc_exp(density_before_activation.to(positions))
            density = density * selector[..., None]
            return density, rgb_embedding

        density, density_embedding = feature2out(feature)
        old_density, old_density_embedding = feature2out(old_feature)
        # block gradients passing through extend loss to the network
        set_requires_grad(self.mlp_base_mlp, False)
        extend_density, extend_density_embedding = feature2out(extend_feature)
        set_requires_grad(self.mlp_base_mlp, True)

        cell_id = cell_id.view(*ray_samples.frustums.shape, -1)
        extend_cell_id = extend_cell_id.view(*ray_samples.frustums.shape, -1)
        importance = importance.view(*ray_samples.frustums.shape, -1)

        if FILTER_OUT_BOX:
            mask = ((positions > 0.25) & (positions < 0.75)).all(dim=-1).view(*ray_samples.frustums.shape, -1)
        else:
            mask = None

        return density, density_embedding, \
                extend_density, extend_density_embedding, \
                old_density, old_density_embedding, \
                cell_id, extend_cell_id, importance, mask

    def get_outputs(
        self, ray_samples: RaySamples, density_embedding: Optional[Tensor] = None
    ) -> Dict[FieldHeadNames, Tensor]:
        assert density_embedding is not None
        outputs = {}
        if ray_samples.camera_indices is None:
            raise AttributeError("Camera indices are not provided.")
        camera_indices = ray_samples.camera_indices.squeeze()
        directions = get_normalized_directions(ray_samples.frustums.directions)
        directions_flat = directions.view(-1, 3)
        d = self.direction_encoding(directions_flat)

        outputs_shape = ray_samples.frustums.directions.shape[:-1]

        h = torch.cat(
            [
                d,
                density_embedding.view(-1, self.geo_feat_dim),
            ],
            dim=-1,
        )
        rgb = self.mlp_head(h).view(*outputs_shape, -1).to(directions)
        outputs.update({FieldHeadNames.RGB: rgb})

        return outputs

    def forward(self, ray_samples: RaySamples, compute_normals: bool = False) -> Dict[FieldHeadNames, Tensor]:
        """Evaluates the field at points along the ray.

        Args:
            ray_samples: Samples to evaluate field on.
        """
        density, density_embedding, \
            extend_density, extend_density_embedding, \
            old_density, old_density_embedding, \
            cell_id, extend_cell_id, importance, mask = self.get_density(ray_samples)

        weight_lower_bound = 1e-7
        if FILTER_OUT_BOX:
            field_outputs = self.get_outputs(ray_samples, density_embedding=density_embedding)
            field_outputs[FieldHeadNames.DENSITY] = torch.where(mask, density, 0) + weight_lower_bound  # type: ignore
            field_outputs[FieldHeadNames.CELLID] = torch.where(mask, cell_id, 0)
            field_outputs[FieldHeadNames.IMPORTANCE] = torch.where(mask, importance, 0)
            field_outputs[FieldHeadNames.RGB] = torch.where(mask.expand(-1,-1,3), field_outputs[FieldHeadNames.RGB], 0)

            # block gradients passing through extend loss to the network
            set_requires_grad(self.mlp_head, False)
            extend_field_outputs = self.get_outputs(ray_samples, density_embedding=extend_density_embedding)
            extend_field_outputs[FieldHeadNames.DENSITY] = torch.where(mask, extend_density, 0) + weight_lower_bound
            extend_field_outputs[FieldHeadNames.RGB] = torch.where(mask.expand(-1,-1,3), extend_field_outputs[FieldHeadNames.RGB], 0)
            extend_field_outputs[FieldHeadNames.CELLID] = torch.where(mask.expand(-1,-1,3), extend_cell_id, 0)
            set_requires_grad(self.mlp_head, True)

            old_field_outputs = self.get_outputs(ray_samples, density_embedding=old_density_embedding)
            old_field_outputs[FieldHeadNames.DENSITY] = torch.where(mask, old_density, 0) + weight_lower_bound
            old_field_outputs[FieldHeadNames.RGB] = torch.where(mask.expand(-1,-1,3), old_field_outputs[FieldHeadNames.RGB], 0)
        else:
            field_outputs = self.get_outputs(ray_samples, density_embedding=density_embedding)
            field_outputs[FieldHeadNames.DENSITY] = density + weight_lower_bound  # type: ignore
            field_outputs[FieldHeadNames.CELLID] = cell_id
            field_outputs[FieldHeadNames.IMPORTANCE] = importance

            # block gradients passing through extend loss to the network
            set_requires_grad(self.mlp_head, False)
            extend_field_outputs = self.get_outputs(ray_samples, density_embedding=extend_density_embedding)
            extend_field_outputs[FieldHeadNames.DENSITY] = extend_density + weight_lower_bound
            extend_field_outputs[FieldHeadNames.CELLID] = extend_cell_id
            set_requires_grad(self.mlp_head, True)

            old_field_outputs = self.get_outputs(ray_samples, density_embedding=old_density_embedding)
            old_field_outputs[FieldHeadNames.DENSITY] = old_density + weight_lower_bound

        return field_outputs, extend_field_outputs, old_field_outputs

    def weight_regularization_loss(self):
        def mlp_regularization_loss(mlp):
            param_list = []
            is_weight = True
            for param in mlp.parameters():
                if is_weight:
                    param_list.append(param)
                    is_weight = False
                else:
                    is_weight = True

            reg_loss = None
            for weight in param_list:
                one_reg_loss = torch.sum(weight**2)
                reg_loss = one_reg_loss if reg_loss==None else (one_reg_loss + reg_loss)

            return reg_loss

        total_reg_loss = mlp_regularization_loss(self.mlp_base_mlp)
        total_reg_loss += mlp_regularization_loss(self.mlp_head)
        return total_reg_loss

    def test_forward(self, step):
        if step == 0:
            return
        device = self.multi_layer_tetra.xyz.device
        rays_o = torch.tensor([-0.05, 0.79, 0.24]).to(device)
        rays_d = torch.tensor([0.85, -0.6, 0.6]).to(device)
        xyz = rays_o[None, :] + torch.linspace(0,1,401).to(device)[112:, None]*rays_d[None, :]
        feature, extend_feature, cell_id, importance = self.multi_layer_tetra.forward(xyz)
        with open(DEBUG_file, "a") as write_file:
            write_file.write(str(cell_id))
            write_file.write(str(feature[:, :2]))

        assert 0