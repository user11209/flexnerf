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

    canonical_cut = torch.gather(canonical, 1, cell_cut)
    children_choice = (canonical_cut[:,0] - canonical_cut[:,1] > 0).int()
    # return shape of (sample_num)
    return children_choice

class MultiLayerTetra(nn.Module):
    def __init__(self,
                aabb,
                feature_dim = 32,
                max_layer_num=8,
                max_cell_count=1e6,
                max_point_count=1e6):
        super().__init__()
        # this list is hard to be reorganized, if using model.load, this list will not reorganize itself, but only the underlying parameters and buffers
        self.register_buffer("aabb", aabb)
        self.register_buffer("feature_dim", torch.tensor(feature_dim))
        self.register_buffer("max_layer_num", torch.tensor(max_layer_num))
        self.register_buffer("max_cell_count", torch.tensor(max_cell_count))
        self.register_buffer("max_point_count", torch.tensor(max_point_count))
        self.register_buffer("first_layer_cell_num", 1)
        assert 2*self.first_layer_cell_num <= max_cell_per_layer, "Variable `max_cell_per_layer` is too small, it at least need to contain all cells on the second layer"

        # !note: cells or points are continuously stored, thus they span a range of cell_offset[i]~cell_offset[i+1]
        #? this is currently not obeyed. all cells and points are not sorted, and we use cell_offset[-1] to find the range
        # **original** cells or points are continuously stored
        self.register_buffer("cell_offset", torch.zeros(max_layer_num+1, dtype=torch.int32))
        self.register_buffer("point_offset", torch.zeros(max_layer_num+1, dtype=torch.int32))
        # **extend** cells or points are continuously stored, 
        # note that the first a few layers are fully covered by original, so extend_offset[i] == extend_offset[i+1] for them
        # extend_offset[0] == 0, and it actually starts from original_offset[-1]
        self.register_buffer("extend_cell_offset", torch.zeros(max_layer_num+1, dtype=torch.int32))
        self.register_buffer("extend_point_offset", torch.zeros(max_layer_num+1, dtype=torch.int32))

        # !note: the following values contain the important properties of the cell pool
        self.register_buffer(
            "child_index",
            -1*torch.ones((self.max_cell_count, 2), dtype=torch.int32)
        )
        self.register_buffer(
            "point_index",
            torch.zeros((self.cell_count, 4), dtype=torch.int32)
        )
        ## child_cut is the way it is sliced into current cell(1 of 6 edges, or 2 of 4 vertices)
        self.register_buffer(
            "child_cut",
            -torch.ones((self.cell_count, 2), dtype=torch.int8)
        )
        self.child_cut[:,1] = 1
        self.register_buffer(
            "parent_index",
            torch.zeros((self.cell_count), dtype=torch.int32)
        )
        ## note: this is the importance of each subdivision candidate. If you want to get the importance of a cell subdivision, one way is to set the importance to be the edge of parent cell's cut (perhaps compare to that without the subdivision afterwards).
        self.register_buffer(
            "importance",
            torch.zeros((self.max_point_count, 6), dtype=torch.float16)
        )
        ## activation_layer stands for the layer where child cell is stored. That is, cells may have a cross-multi-layer parent-child relationship. 
        ## note: A cell is only activated when seach_layer_i(i == activation_layer)
        ## activation_layer in range(0, max_layer_num) for original; range(max_layer_num, 2*max_layer_num) for extend
        self.register_buffer(
            "activation_layer",
            -torch.ones(self.max_cell_count, dtype=torch.int8)
        )
        ## layer is where the cell is stored, in another way, `parent.activation_layer == child.layer`
        self.register_buffer(
            "layer",
            -torch.ones(self.max_cell_count, dtype=torch.int8)
        )
        ## sync_to and sync_to_trigger is for synchronize_cell2edge, representing how to transfer information between edges which concide with each other but are stored in different cells
        self.register_buffer(
            "sync_to_trigger",
            -torch.ones(self.max_cell_count, 6, dtype=torch.int8)
        )
        self.register_buffer(
            "sync_to",
            -torch.ones(2, self.max_cell_count, 6, dtype=torch.int32)
        )
        ## neighbor means the face neighbor of the cell, each cell have 4 neighbor. it is like a face-wise `sync_to`
        self.register_buffer(
            "neighbor",
            -torch.ones(2, self.max_cell_count, 4, dtype=torch.int32)
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
        self.register_buffer("syncing_sorted", torch.tensor([False]))
        self.register_buffer("edge2point", torch.tensor([[0,1],[0,2],[0,3],[1,2],[1,3],[2,3]], dtype=torch.int8))
        self.register_buffer("point2edge", torch.tensor([[-1,0,1,2], [0,-1,3,4], [1,3,-1,5], [2,4,5,-1], dtype=torch.int8))
        self.register_buffer("face2point", torch.tensor([[1,2,3],[0,2,3],[0,1,3],[0,1,2]], dtype=torch.int8))

        # !note: registering stop here, organize the first layer of tree and try sampling
        self.organize_field_0()
        self.refine_samples()

    def refine_samples(self, update_original):
        self.remove_last_sampled()
        if update_original:
            self.merge_extend()
        edge_valid = self.sample_extend()
        self.apply_sampled(edge_valid)

    def forward(self, xyz):
        cells = self.search_field_0(xyz)
        for i in range(1, max_layer_num-1):
            cells = self.search_layer_i(xyz, cells, i))
        if self.mode == "extend":
            cells = self.search_extend(xyz, cells)
        feature = interpolate_field_value(xyz, cells)
        return feature

    # ! detailed function definition starts here
    def organize_layer_0(self):
        self.set_layer_cell_size(0,1, mode="original")
        self.set_layer_point_size(0,4, mode="original")
        self.point_index[:1,:] = torch.arange(4, device=self.point_index.device).view(1,4)
        self.layer[:1] = 0
        self.sync_to_trigger[:1] = 0

        point_xyz = self.get_tetra_xyz(aabb)
        self.xyz[:4,:] = point_xyz

        #? temporarily use cell_offset[-1] instead of cell_offset[1], that is, the cells and points are not sorted
        self.cell_offset[0] = 0
        self.cell_offset[-1] = 1
        self.point_offset[0] = 0
        self.point_offset[-1] = 4
    
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
        cells = self.search_layer_i(xyz, cells, max_layer_num))
        return cells

    def remove_last_sampled(self):
        # TODO
        pass

    def merge_extend(self, merged_cell_count):
        with torch.inference_mode():
            is_leaf_cell = self.is_leaf_cell().view(-1,1).expand(-1,6)
            def merge_to_parent_importance(main, mirror):
                # TODO: define how different importance values stored for the same edge are combined
                pass
            self.synchronize_cell2edge(self.importance, merge_to_parent_importance)

            valid_edge_importance = self.importance*is_leaf_cell
            _, edge_chosen_index_1d = torch.sort(valid_edge_importance.view(-1))
            edge_chosen_id_1d = edge_chosen_id_1d[:merged_cell_count]
            edge_chosen = torch.zeros(self.max_cell_count, 6, dtype=torch.bool)
            torch.scatter_(edge_chosen.view(-1), 0, edge_chosen_index_1d, torch.ones_like(edge_chosen_index_1d, dtype=torch.bool))

            def choose_edge_with_highest_importance(edge_valid, cell_edge_num, cell_chosen):
                # TODO: choose the highest importance from all valid edges
                pass

            edge_valid_for_original = self.broadcast_n_remove_conflict(edge_chosen, is_leaf_cell, choose_edge_with_highest_importance)

            self.apply_sampled(edge_valid_for_original)

            #! start setting structural variables
            child_start = self.cell_offset[-1]
            child_end = self.cell_offset[-1] + self.extend_cell_offset[-1]
            parent_cell_index = self.parent_index[child_start:child_end]
            parent_child_cut = self.child_cut[parent_cell_index, :]
            parent_neighbor = self.neighbor[:, parent_cell_index, :]

            child_index_relative = torch.arange(child_end - child_start, dtype=torch.int32, device=self.cell_offset.device)
            child_01 = child_index_relative % 2
            child_index = child_index_relative + child_start
            abandoned_vertex = torch.gather(parent_child_cut, 1, (1-child_01).view(-1,1))
            maintained_vertex = torch.gather(parent_child_cut, 1, child_01.view(-1,1))
            # **set child neighbor
            child_neighbor = parent_neighbor.clone()

            # if the old neighbor has children, set new neighbor to be the children of the old neighbors
            neighbor_cell_id = child_neighbor[0,:,:]
            neighbor_has_child = (self.child_index[neighbor_cell_id, 0] != -1)
            neighbor_child_cut_0 = self.child_cut[neighbor_cell_id, 0]
            neighbor_child_0_isnewneighbor = (child_neighbor[1,:,:] != neighbor_child_cut_0)
            neighbor_child_id = (~neighbor_child_0_isnewneighbor).int()
            substitute_neighbor_id = self.child_index[neighbor_cell_id, neighbor_child_id]
            child_neighbor[0,:,:] = torch.where(neighbor_has_child, substitute_neighbor_id, child_neighbor[0,:,:])

            # one neighbor of a child cell is set to be the alternative child cell
            abandoned_face = maintained_vertex.view(-1,1)
            substitute_face_cellnum = (child_index_relative//2*2+1-child_01+child_start).view(-1,1)
            substitute_face_facenum = abandoned_vertex.view(-1,1)
            child_neighbor[0,:,:].scatter_(1, abandoned_face, substitute_face_cellnum)
            child_neighbor[1,:,:].scatter_(1, abandoned_face, substitute_face_facenum)
            
            # TODO: set self.sync_to_trigger and self.sync_to
            parent_sync_to = self.sync_to[:, parent_cell_index, :]
            parent_sync_to_trigger = self.sync_to_trigger[parent_cell_index, :]

            child_sync_to = torch.stack((parent_cell_index.view(-1,1).expand(-1,6), \
                        torch.arange(6, device=parent_cell_index.device).view(1,6).expand(child_end-child_start, 6)), axis=0)
            # get special edge which concides with child_cut
            # get special edges connecting `abandoned_vertex` and two uninfluenced vertices
            # TODO: temporarily only set the values of the activation layers

    def sample_extend(self, sample_value_weight, sample_threshold):
        with torch.inference_mode():
            cell_num = self.cell_offset[-1]
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
                self.synchronize_cell2edge(edge_chosen, torch.logical_or)
                edge_valid = self.broadcast_n_remove_conflict(edge_chosen, is_leaf_cell, self.choose_edge)

                cell_edge_num = torch.sum(edge_valid, axis=1)
                if torch.all(torch.logical_or(cell_edge_num <= 1, 1-is_leaf_cell)):
                    break
        
        return edge_valid

    def apply_sampled(self, edge_valid):
        with torch.inference_mode():
            extend_offset = self.cell_offset[-1]
            is_leaf_cell = self.is_leaf_cell().view(-1,1).expand(-1,6)
            edge_valid_leaf = torch.logical_and(is_leaf_cell, edge_valid)
            # **each edge which is valid for subdivision corresponds to a point in the extend layer, we need to get each edge an index(the same edges in different cells correspond to the same point index).
            # set each unique edge a number and remove the conflicts
            trace_edge = torch.arange(6*self.cell_count, device=edge_valid.device).view(self.cell_count, 6) + 1
            self.synchronize_cell2edge(trace_edge, torch.max)
            # each valid leaf cell contains an edge for subdivision, convert the edge number to the point index causing the subdivision
            trace_edge_in_cell = torch.sum(edge_valid*trace_edge, axis=1)
            sorted_trace_edge_in_cell, sort_index = torch.sort(trace_edge_in_cell, descending=False)
            step_posi = sorted_trace_edge_in_cell[:-1] - sorted_trace_edge_in_cell[1:]
            head = torch.tensor([0], dtype=torch.int32, device=step_posi.device)
            sorted_trace_edge_in_cell_normalized = torch.concatenate((head, torch.cumsum(step_posi != 0, dim=0)))
            # get "which cell index is divided by which point index & which point can be represented by which cell"
            parent_cell_num = torch.sum(edge_valid_leaf)
            parent_cell_index = sort_index[:parent_cell_num]
            subdivision_point_index = sorted_trace_edge_in_cell_normalized[:parent_cell_num] + self.point_offset[-1]

            one_cell_per_point_mid_index = torch.nonzero(step_posi).view(-1)
            one_cell_per_point = sort_index[one_cell_per_point_mid_index]

            # **connect parent and children
            parent_child_index = torch.arange(2*parent_cell_num, device=edge_valid.device).view(-1,2) + self.cell_offset[-1]

            parent_edge_index = torch.argmax(edge_valid[parent_cell_index, :], axis=1)
            parent_child_cut = self.edge2point[parent_edge_index, :]

            self.child_index[parent_cell_index, :] = parent_child_index
            self.child_cut[parent_cell_index, :] = parent_child_cut
            self.activation_layer[parent_cell_index] = self.max_layer_num

            # **construct children cells
            children_parent_index = parent_cell_index.view(-1,1).expand(-1,2)

            abandoned_vertex = torch.stack((parent_child_cut[:,1], parent_child_cut[:,0]), axis=1).view(-1,2,1)
            parent_point_index = self.point_index[parent_cell_index, :]
            children_point_index = parent_point_index.view(-1,1,4).expand(-1,2,4) \
                                .scatter(2, abandoned_vertex, subdivision_point_index.view(-1,1).expand(-1,2))

            child_start = self.cell_offset[-1]
            child_end = self.cell_offset[-1] + 2*parent_cell_num
            self.parent_index[child_start:child_end] = children_parent_index.view(-1)
            self.point_index[child_start:child_end, :] = child_point_index.view(-1,4)
            self.child_index[child_start:child_end, :] = -1
            self.child_cut[child_start:child_end, :] = -1
            self.activation_layer[child_start:child_end] = -1
            self.layer[child_start:child_end] = self.max_layer_num
            # TODO: MAX_SYNC_TO_TRIGGER not defined
            self.sync_to_trigger[child_start:child_end] = MAX_SYNC_TO_TRIGGER
            self.sync_to[:, child_start:child_end, :] = -1

            # **construct children points
            divided_point_index_in_cell = parent_child_cut[one_cell_per_point_mid_index, :]
            divided_point_index = torch.gather(parent_point_index, 1, divided_point_index_in_cell)
            child_field = torch.sum(self.field[divided_point_index, :], axis=1) / 2
            child_xyz = torch.sum(self.xyz[divided_point_index, :], axis=1) / 2

            point_start = self.point_offset[-1]
            point_end = self.point_offset[-1] + one_cell_per_point_mid_index.shape[0]
            self.field[point_start:point_end, :] = child_field
            self.xyz[point_start:point_end, :] = child_xyz

            # **change overall extend label
            self.extend_cell_offset[-1] = child_end - child_start
            self.extend_point_offset[-1] = point_end - point_start

    # ! second level detailed function definition starts here
    def get_tetra_xyz(self, aabb):
        #TODO: get the tetrahedron cover of `aabb`, only xyz values, so shape (4,3)
        pass

    def is_leaf_cell(self):
        # TODO: mistaken `is_leaf_cell` condition, self.activation_layer should be larger than self.max_layer_num? to exclude extend layer. (but it might exclude original leaf cell with no extend child, we may need to fix this)
        return torch.logical_or(self.activation_layer>=self.max_layer_num, self.activation_layer==-1)

    def set_layer_cell_size(self, layer_i, size, mode):
        if mode == "original":
            orig_size = self.cell_offset[layer_i+1] - self.cell_offset[layer_i]
            self.cell_offset[layer_i+1:] += (size - orig_size)
        elif mode == "extend":
            #? this is not needed, remove it later
            orig_size = self.extend_cell_offset[layer_i+1] - self.extend_cell_offset[layer_i]
            self.extend_cell_offset[layer_i+1:] += (size - orig_size)
        else:
            import sys
            sys.exit("there should be no other modes!")
        
    def set_layer_point_size(self, layer_i, size, mode):
        if mode == "original":
            orig_size = self.point_offset[layer_i+1] - self.point_offset[layer_i]
            self.point_offset[layer_i+1:] += (size - orig_size)
        elif mode == "extend":
            orig_size = self.extend_point_offset[layer_i+1] - self.extend_point_offset[layer_i]
            self.extend_point_offset[layer_i+1:] += (size - orig_size)
        else:
            import sys
            sys.exit("there should be no other modes!")

    def choose_edge(edge_valid, cell_edge_num, cell_chosen):
        cell_num = edge_valid.shape[0]
        edge_chosen_index_in_valid = ( torch.rand(cell_num) * cell_edge_num ).int() + 1
        confirmed = torch.zeros(cell_num, dtype=torch.bool, device=edge_valid.device)
        edge_chosen_index = torch.zeros(cell_num, dtype=torch.int8, device=edge_valid.device)
        edge_chosen = torch.zeros(cell_num, 6, dtype=torch.bool, device=edge_valid.device)
        valid_edge_count_prefix = torch.cumsum(edge_valid, dim=1)
        for i in range(6):
            chosen_condition = torch.logical_and(valid_edge_count_prefix[:,i]==edge_chosen_index_in_valid, 1-confirmed)
            confirmed = torch.logical_or(chosen_condition, confirmed)
            edge_chosen_index = torch.where(chosen_condition, i, edge_chosen_index)

        edge_chosen = torch.scatter(edge_chosen, 1, edge_chosen_index.view(-1,1), cell_chosen)
        return edge_chosen_index, edge_chosen

    def broadcast_n_remove_conflict(edge_chosen, is_leaf_cell, choose_edge_func):
        # **conflictions may happen in the following scenario: a and b choose two different common edges with c. To fix this, choose only one edge in each conflicting cell and invalidate others. if an edge is at the same time labeled `invalid` and `not invalid`, invalidate it.
        # note, cells with only one valid edge never get invalidated, because all conflicting edges have been invalidated by the next step in the last round, and will not be chosen this round
        cell_chosen_edge_num = torch.sum(edge_chosen, axis=1)
        cell_related = torch.logical_and(cell_chosen_edge_num != 0, is_leaf_cell)
        edge_chosen_index, edge_chosen = choose_edge_func(edge_chosen, cell_chosen_edge_num, cell_related)
        self.synchronize_cell2edge(edge_chosen, torch.logical_and)
                
        cell_chosen = (torch.sum(edge_chosen, axis=1) == 1)
        # **In each cell, if an edge is chosen, all other edges are invalidated. if an edge is at the same time labeled `invalid` and `not invalid`, invalidate it.
        edge_valid = (1-cell_chosen).view(-1,1).expand(-1,6)
        torch.scatter_(edge_valid, 1, edge_chosen_index, torch.ones_like(cell_chosen))
        self.synchronize_cell2edge(edge_valid, torch.logical_and)

        return edge_valid

    def synchronize_cell2edge(sync_src, reduction_func):
        # synchronize the value of each edge,  because they(`sync_src`) are stored and updated at the structure of cells. do reduction along the tree to a partial root, and then spread the reduction results back to the leaves
        sync_buffer = torch.zeros_like(sync_src)
        sync_label = torch.zeros_like(sync_src)
        sync_label_true = torch.ones_like(sync_src)

        self.sort_for_syncing()
        # TODO: self.sync_to_trigger, self.sync_to, maximum is not defined
        #? sync_src.copy_ give a costly implementation, change it into a cheaper one if necessary
        for trigger in range(maximum-1, 0, -1):
            is_syncing_to_cellnum, is_syncing_to_edgenum = self.get_sync_trigger_index(trigger)
            self.write_src_to_dst(sync_src, sync_buffer, self.sync_to, is_syncing_to_cellnum, is_syncing_to_edge_num)
            self.write_src_to_dst(sync_label_true, sync_label, self.sync_to, is_syncing_to)
            update_value = reduction_func(sync_src, sync_buffer)
            sync_src.copy_(torch.where( sync_label, update_value, sync_src ))
            sync_label[:] = 0
        
        for trigger in range(1, maximum, 1):
            is_syncing_from_cellnum, is_syncing_from_edgenum = self.get_sync_trigger_index(trigger)
            self.read_dst_to_src(sync_src, sync_buffer, self.sync_to, is_syncing_from_cellnum, is_syncing_from_edgenum)
            self.read_dst_to_src(sync_label_true, sync_label, self.sync_to, is_syncing_from_cellnum, is_syncing_from_edgenum)
            sync_src.copy_(torch.where( sync_label, sync_buffer, sync_src ))
            sync_label[:] = 0

    # ! third level detailed function definition starts here
    def sort_for_syncing():
        #? still lack checking for invalid slots, that is, for `index >= self.max_cell_count`, we still have to manually set their `sync_to_trigger` values to a preset large constant value.
        #? memorize to change self.syncing_sort to False after subdivision
        if not self.syncing_sorted:
            sorted_sync_to_trigger, sort_sync_to_trigger_index_1d = torch.sort(self.sync_to_trigger.view(-1))
            self.sort_sync_to_trigger_cellnum = sort_sync_to_trigger_index_1d // 6
            self.sort_sync_to_trigger_edgenum = sort_sync_to_trigger_index_1d %  6
            step_posi = sorted_sync_to_trigger[1:] - sort_sync_to_trigger[:-1]
            head = torch.tensor([0], dtype=torch.int32, device=step_posi.device)
            self.sorted_trigger_offset = torch.concatenate((head, 1+torch.nonzero(step_posi).view(-1)))
            self.syncing_sorted[0] = True

    def get_sync_trigger_index(trigger):
        sorted_start_index = self.sorted_trigger_offset[trigger]
        sorted_end_index = self.sorted_trigger_offset[trigger+1]
        return self.sync_to_trigger_cellnum[sorted_start_index: sorted_end_index], \
               self.sync_to_trigger_edgenum[sorted_start_index: sorted_end_index]

    def write_src_to_dst(src, dst, src_to_dst, src_index_cellnum, src_index_edgenum):
        #? this is a waste, because it is just calculated in the above function `sort_for_syncing`
        src_index_1d = 6*src_index_cellnum + src_index_edgenum
        value = torch.gather(src.view(-1), 0, src_index_1d)
        dst_index_cell = torch.gather(src_to_dst[0,:,:].view(-1), 0, src_index_1d)
        dst_index_edge = torch.gather(src_to_dst[1,:,:].view(-1), 0, src_index_1d)
        dst_index_1d = 6*dst_index_cell + dst_index_edge
        dst.view(-1).scatter_(0, dst_index_1d, value)

    def read_dst_to_src(src, dst, src_to_dst, src_index_cellnum, src_index_edgenum):
        #? this is a waste, because it is just calculated in the above function `sort_for_syncing`
        src_index_1d = 6*src_index_cellnum + src_index_edgenum
        dst_index_cell = torch.gather(src_to_dst[0,:,:].view(-1), 0, src_index_1d)
        dst_index_edge = torch.gather(src_to_dst[1,:,:].view(-1), 0, src_index_1d)
        dst_index_1d = 6*dst_index_cell + dst_index_edge
        value = torch.gather(src.view(-1), 0, dst_index_1d)
        dst.view(-1).scatter_(0, src_index_1d, value)

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

        self.multi_layer_tetra = MultiLayerTetra(aabb, feature_dim, max_layer_num, max_point_per_layer)

        self.mlp_base_mlp = MLP(
            in_dim=self.feature_dim,
            num_layers=num_layers,
            layer_width=hidden_dim,
            out_dim=1 + self.geo_feat_dim,
            activation=nn.ReLU(),
            out_activation=None,
            implementation=implementation,
        )
        self.mlp_base = torch.nn.Sequential(self.multi_layer_tetra, self.mlp_base_mlp)

        self.mlp_head = MLP(
            in_dim=self.direction_encoding.get_out_dim() + self.geo_feat_dim,
            num_layers=num_layers_color,
            layer_width=hidden_dim_color,
            out_dim=3,
            activation=nn.ReLU(),
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
        h = self.mlp_base(positions_flat).view(*ray_samples.frustums.shape, -1)
        # print("___________profiler___________")
        # print("shape for positions_flat is ", positions_flat.shape, ", and for h is ", h.shape)
        density_before_activation, base_mlp_out = torch.split(h, [1, self.geo_feat_dim], dim=-1)
        self._density_before_activation = density_before_activation

        # Rectifying the density with an exponential is much more stable than a ReLU or
        # softplus, because it enables high post-activation (float32) density outputs
        # from smaller internal (float16) parameters.
        density = trunc_exp(density_before_activation.to(positions))
        density = density * selector[..., None]
        return density, base_mlp_out

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
                density_embedding.view(-1, self.geo_feat_dim)
            ],
            dim=-1,
        )
        rgb = self.mlp_head(h).view(*outputs_shape, -1).to(directions)
        outputs.update({FieldHeadNames.RGB: rgb})

        return outputs
