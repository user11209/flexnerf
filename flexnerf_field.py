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

def to_canonical(xyz, cell_xyz):
    xyz: Tensor[sample_num, 3]
    cell_xyz: Tensor[sample_num, 4, 3]
    canonical: Tensor[sample_num, 4]

    # [a, b, c, d] @ convert_matrix = [x, y, z, 1]
    convert_matrix = torch.transpose( torch.concatenate(cell_xyz, torch.ones_like(xyz).view(-1,3,1), axis=-1), 1,2)
    convert_xyz = torch.concatenate(xyz, torch.ones_like(xyz[:,:1]), axis=-1)
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
    interpolated_feature = torch.sum(cell_feature * canonical, axis=1)
    if len(xyz.shape) == 3:
        interpolated_feature = interpolated_feature.view(xyz.shape[0], xyz.shape[1], -1)
    return interpolated_feature

def divide_into_children(xyz, cells):
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

    canonical = to_canonical(xyz, cell_xyz)

    canonical_cut = torch.gather(canonical, 1, cell_cut.long())
    children_choice = (canonical_cut[:,0] - canonical_cut[:,1] > 0).long()
    # return shape of (sample_num)
    return children_choice

class MultiLayerTetra(nn.Module):
    def __init__(self,
                aabb,
                feature_dim = 32,
                max_layer_num=20,
                max_cell_count=8e5,
                max_point_count=1e5):
        super().__init__()
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
        self.register_buffer("cell_offset", torch.zeros(1, dtype=torch.int64))
        self.register_buffer("edge_offset", torch.zeros(1, dtype=torch.int64))
        self.register_buffer("point_offset", torch.zeros(1, dtype=torch.int64))
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
            torch.zeros((self.max_cell_count, 4), dtype=torch.int64)
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
        ## note: this is the importance of each subdivision candidate. If you want to get the importance of a cell subdivision, one way is to set the importance to be the edge of parent cell's cut (perhaps compare to that without the subdivision afterwards).
        self.register_buffer(
            "importance",
            torch.zeros((self.max_cell_count, 6), dtype=torch.float32)
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

        # !note: the following values contain some flags valueable in tree organization
        self.register_buffer("edge_sorted", torch.tensor([False]))

        # !note: registering stop here, organize the first layer of tree and try sampling
        self.organize_layer_0()
        self.refine_samples(True)

    def refine_samples(self, update_original):
        self.remove_last_sampled()
        # TODO: check boundaries, no to exceed the `max_??_count`, especially for sample_extend. Note, merge_extend lack 1 positional argument for that reason. temporarily set it to be 2. Note, this value means how many parent cells will be divided, so the actual increased cell number is at most twice the value
        if update_original:
            self.merge_extend(torch.tensor(2, dtype=torch.int64))
        edge_valid = self.sample_extend()
        self.apply_sampled(edge_valid)

    def forward(self, xyz):
        cells = self.search_field_0(xyz)
        for i in range(1, self.max_layer_num-1):
            cells = self.search_layer_i(xyz, cells, i)
        if self.mode == "extend":
            cells = self.search_extend(xyz, cells)
        feature = interpolate_field_value(xyz, cells)
        return feature

    # ! detailed function definition starts here
    def organize_layer_0(self):
        self.point_index[:1,:] = torch.arange(4, device=self.point_index.device).view(1,4)
        self.edge_index[:1] = torch.arange(6, device=self.edge_index.device).view(1,6)
        self.layer[:1] = 0

        point_xyz = self.get_tetra_xyz(self.aabb)
        self.xyz[:4,:] = point_xyz

        self.cell_offset[0] = 1
        self.edge_offset[0] = 6
        self.point_offset[0] = 4
    
    def search_layer_0(self, xyz):
        sample_size = xyz.shape[0]

        cell_id = torch.zeros(sample_size, dtype=torch.int32, device=xyz.device)
        cell_xyz = self.xyz[:4, :].view(1,4,3).expand(sample_size, 4, 3)
        cell_feature = self.field[:4, :].view(1,4,-1).expand(sample_size, 4, -1)

        cut = self.child_cut[:1, :].view(1, 2).expand(sample_size, 2)
        activation_layer = self.activation_layer[:1].expand(sample_size)

        return {"cell_id":          cell_id, 
                "xyz":              cell_xyz, 
                "feature":          cell_feature, 
                "cut":              cut, 
                "activation_layer": activation_layer}

    def search_layer_i(self, xyz, parent_cells, i):
        sample_size = xyz.shape[0]

        parent_id = parent_cells["cell_id"]
        parent_xyz = parent_cells["xyz"]
        parent_feature = parent_cells["feature"]
        parent_cut = parent_cells["cut"]
        parent_activation_layer = torch.min(parent_cells["activation_layer"], self.max_layer_num)

        children_choice = divide_into_children(xyz, parent_cells)
        # **child_cell_id
        child_cell_id = self.child_index[parent_id, children_choice]
        # **child_valid, mid variable
        child_valid = torch.logical_and(parent_id != -1, child_cell_id != -1)
        child_valid = torch.logical_and(child_valid, parent_activation_layer == i)
        # **child_vertex_id, mid variable, now represented as abandoned_vertex & child_vertex_id_substitute
        # only one vertex of the parent cell will change when divided into a subcell
        ## element of abandoned_vertex is in 0,1,2,3
        abandoned_vertex = torch.gather(parent_cut, 1, 1-children_choice.view(-1,1))
        child_vertices_id_substitute = self.point_index[child_cell_id, abandoned_vertex]
        # **child_feature
        #? no longer needed, remove it later
        child_feature = parent_feature.clone()
        child_feature_substitute = self.field[child_vertices_id_substitute, :]
        child_feature[torch.arange(sample_num), abandoned_vertex, :] = child_feature_substitute
        # **child_xyz
        child_xyz = parent_xyz.clone()
        cut_point_xyz = parent_xyz[torch.arange(sample_num).view(-1,1), cell_cut, :]
        child_xyz_substitute = torch.sum(cut_point_xyz, axis=1) / 2
        child_xyz[torch.arange(sample_num), abandoned_vertex, :] = child_xyz_substitute
        # **child_cut
        child_cut = self.child_cut[child_cell_id, :]
        # **child_activation_layer
        child_activation_layer = self.activation_layer[child_cell_id]

        # filter out inactivated cells
        child_cell_id          = torch.where(child_valid,            child_cell_id,          parent_id)
        child_feature          = torch.where(child_valid.view(-1,1), child_feature,          parent_feature)
        child_xyz              = torch.where(child_valid.view(-1,1), child_xyz,              parent_xyz)
        child_cut              = torch.where(child_valid.view(-1,1), child_cut,              parent_cut)
        child_activation_layer = torch.where(child_valid,            child_activation_layer, parent_activation_layer)

        return {"cell_id":          child_cell_id, 
                "xyz":              child_cell_xyz, 
                "feature":          child_cell_feature, 
                "cut":              child_cut, 
                "activation_layer": child_activation_layer}

    def search_extend(self, xyz):
        cells = self.search_layer_i(xyz, cells, self.max_layer_num)
        return cells

    @torch.inference_mode()
    def remove_last_sampled(self):
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

        #** clean up overall extend labels
        self.extend_cell_offset[0] = 0
        self.extend_point_offset[0] = 0

    @torch.inference_mode()
    def merge_extend(self, merged_cell_count):
        is_leaf_cell = self.is_leaf_cell()
        #? use a more elegant function for importance synchronization, rather than "average"
        self.importance.copy_(self.synchronize_cell2edge(self.importance, "average"))

        #? number of chosen edges is fixed to be merged_cell_count, but each cell may at most include 6 chosen edges, so the number of actual chosen cell falls far below merged_cell_count.
        valid_edge_importance = self.importance*is_leaf_cell.view(-1,1)
        #? valid_edge_importance will not show validity if self.importance=0 for valid points. We thus use `stable=True` here to temporarily avoid this mess. another way might be to use different importance values, this might be cheaper and more robust.
        sorted_valid_edge_importance, edge_candidate_index_1d = torch.sort(valid_edge_importance.view(-1), stable=True)
        edge_candidate_index_1d = edge_candidate_index_1d[:merged_cell_count]
        edge_candidate = torch.zeros(self.max_cell_count, 6, dtype=torch.bool)
        edge_candidate.view(-1).scatter_(0, edge_candidate_index_1d, torch.ones_like(edge_candidate_index_1d, dtype=torch.bool))

        cell_candidate = (torch.sum(edge_candidate, axis=1) != 0)
        # the following line is only to remove the edge candidate with the least importance, which may create more subcells than expected
        edge_candidate = self.synchronize_cell2edge(edge_candidate, "and")
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
            sample_rate = torch.rand(self.max_cell_count,6) * torch.max(sample_value) * 10/(iteration+10)

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
            edge_determined = torch.logical_and(edge_valid_temp, chosen_edge_stage2)

            candidate_edge_num = torch.sum(edge_candidate, axis=1)
            if torch.all(candidate_edge_num <= 1):
                break
        # note, return of broadcast_n_remove_conflict only determines the `invalid`s, so only cells with a fixed chosen edge should be taken into account
        chosen_cell = (torch.sum(edge_candidate, axis=1) == 1)
        edge_valid_for_original = edge_candidate
        #? .int() is for argmax_cpu, and might be removed after debugging
        edge_valid_for_original_idincell = torch.argmax(edge_valid_for_original.int(), axis=1)

        self.apply_sampled(edge_valid_for_original)

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
        cell_valid_for_original = torch.any(edge_valid_for_original, axis=1)
        cell_activation_layer = torch.sum(edge_layer*edge_valid_for_original, axis=1).int()
        self.activation_layer = torch.where(cell_valid_for_original, cell_activation_layer, self.activation_layer)
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
        cell_num = self.max_cell_count
        is_leaf_cell = self.is_leaf_cell()
        # the following variables are only valid for condition `is_leaf_cell`, that is, only leaf of the tree can choose edges.
        edge_valid = torch.ones(self.max_cell_count, 6)
        cell_edge_num = 6*torch.ones_like(self.activation_layer)
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
        divided_point_index_in_cell = self.child_cut[one_cell_per_point, :].long()
        divided_point_index = torch.gather(parent_point_index, 1, divided_point_index_in_cell)
        child_field = torch.sum(self.field[divided_point_index, :], axis=1) / 2
        child_xyz = torch.sum(self.xyz[divided_point_index, :], axis=1) / 2

        point_start = self.point_offset[0]
        point_end = self.point_offset[0] + one_cell_per_point.shape[0]
        self.field[point_start:point_end, :] = child_field
        self.xyz[point_start:point_end, :] = child_xyz

        # **change overall extend label
        self.extend_cell_offset[0] = child_end - child_start
        self.extend_point_offset[0] = point_end - point_start

    # ! second level detailed function definition starts here
    def get_tetra_xyz(self, aabb):
        #TODO: get the tetrahedron cover of `aabb`, only xyz values, so shape (4,3)
        return torch.tensor([[0,0,0], [0,0,1], [0,1,0], [1,0,0]], dtype=torch.float32)

    def is_leaf_cell(self):
        is_leaf_candidate = torch.logical_or(self.activation_layer>=self.max_layer_num, self.activation_layer==-1)
        is_original_cell = (torch.arange(self.max_cell_count, device=self.activation_layer.device) < self.cell_offset[0])
        is_leaf_cell = torch.logical_and(is_leaf_candidate, is_original_cell)
        return is_leaf_cell

    def choose_edge(self, edge_valid, cell_edge_num, cell_chosen):
        cell_num = edge_valid.shape[0]
        edge_chosen_index_in_valid = ( torch.rand(cell_num) * cell_edge_num ).int() + 1
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
        edge_chosen = self.synchronize_cell2edge(edge_chosen, "and")
                
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
        #? the following line might be deleted when integrated
        sync_src = sync_src.int()
        if reduction_func == "max":
            buffer = torch.zeros_like(sync_src)
            
            max_cell2sortededge_map_inner = torch.argsort(sync_src.view(-1), descending=True)
            sorted_edge_index_inner = self.edge_index.view(-1)[max_cell2sortededge_map_inner]
            sorted_edge_index, max_cell2sortededge_map_outter = torch.sort(sorted_edge_index_inner, stable=True)
            max_cell2sortededge_map = max_cell2sortededge_map_outter[max_cell2sortededge_map_inner]
            sync_src_indexsort = sync_src.view(-1)[max_cell2sortededge_map]

            sorted_edge_changing = (sorted_edge_index[1:]-sorted_edge_index[:-1] != 0)
            head = torch.tensor([0], dtype=sorted_edge_index.dtype, device=sorted_edge_index.device)
            leading_edge = torch.concatenate((head, 1+torch.nonzero(sorted_edge_changing).view(-1)), axis=0)
            leadedby_edge = torch.concatenate((head, torch.cumsum(sorted_edge_changing, dim=0)), axis=0)
            sorted_max_edge = leading_edge[leadedby_edge]
            sorted_max_sync_src = sync_src_indexsort[sorted_max_edge]

            torch.scatter(buffer.view(-1), 0, max_cell2sortededge_map, sorted_max_sync_src)
            return buffer

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
            groupby_result = torch.arange(groupby_sum.shape[0], device=groupby_sum.device)
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

        twolevel_sorting_indices = outer_sorting_indices[inner_sorting_indices]
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
        twolevel_sorted_offset = leading_subedge[leadedby_subedge]
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
        leading_subedge_point_index = twolevel_sorted_selected_point[leading_subedge]
        twolevel_sorted_issubedge = (leading_subedge_point_index != twolevel_sorted_selected_point)
        ret_cellindex_selected = torch.scatter(selected_point_index.view(-1), 0, \
                                    twolevel_sorting_indices, twolevel_sorted_issubedge.long()).view(-1,2)

        full_subedge_count = 1+torch.sum(twolevel_sorted_changing)

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
        if not self.edge_sorted:
            sorted_edge_index, self.cell2sortededge_map = torch.sort(self.edge_index.view(-1), descending=True)
            self.sorted_edge_index = sorted_edge_index
            offset = torch.nonzero(sorted_edge_index[1:] - sorted_edge_index[:-1]).view(-1)
            head = torch.zeros([1], dtype=offset.dtype, device=offset.device)
            # please note, `sorted_edge_index[offset[-1]:] == -1` necessarily happens, and the element number is not 0. that is why we do not take these elements into account in the following.
            self.sorted_edge_offset = torch.concatenate((head, offset+1), axis=0)
            self.sorted_edge_index_unique = sorted_edge_index[offset]
            self.edge_sorted[0] = True

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
        max_layer_num: int = 1,
        max_point_per_layer: int = 1e5,
        num_layers: int = 6,
        hidden_dim: int = 64,
        geo_feat_dim: int = 15,
        num_layers_color: int = 3,
        hidden_dim_color: int = 64,
        spatial_distortion: Optional[SpatialDistortion] = None,
        implementation: Literal["tcnn", "torch"] = "tcnn",
    ) -> None:
        super().__init__()
        self.register_buffer("aabb", aabb)
        self.feature_dim = feature_dim

        self.geo_feat_dim = geo_feat_dim
        self.spatial_distortion = spatial_distortion

        self.direction_encoding = SHEncoding(
            levels=4,
            implementation=implementation,
        )

        self.multi_layer_tetra = MultiLayerTetra(aabb, feature_dim, max_layer_num, 8*max_point_per_layer, max_point_per_layer)