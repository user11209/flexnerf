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

def interpolate_field_value(xyz, cells):
    xyz: Tensor[sample_num, each_cell_num, 3] or Tensor[sample_num, 3]
    cell_xyz: Tensor[sample_num, 8, 3]

    xyz_shape = xyz.shape
    if len(xyz_shape) == 3:
        xyz = xyz.view(-1,3)
        cell_xyz = cells["xyz"].unsqueeze(1).expand(-1, xyz_shape[1], -1, -1).view(-1, 8, 3)
        cell_feature = cells["feature"].unsqueeze(1).expand(-1, xyz_shape[1], -1, -1).view(xyz_shape[0]*xyz_shape[1], 8, -1)
    else:
        cell_xyz = cells["xyz"]
        cell_feature = cells["feature"]

    #TODO: trilinear interpolation, should be later changed to more subtule tetrahedron interpolation
    canonical = (xyz - cell_xyz[:,0,:]) / (cell_xyz[:,7,:] - cell_xyz[:,0,:])
    canonical_contrast = torch.stack((1-canonical, canonical), 1) # Tensor[sample_num, 2, 3]
    canonical_factor = (canonical_contrast[:,:,0].unsqueeze(-1).unsqueeze(-1).expand(-1, 2, 2, 2) *\
                        canonical_contrast[:,:,1].unsqueeze( 1).unsqueeze(-1).expand(-1, 2, 2, 2) *\
                        canonical_contrast[:,:,2].unsqueeze( 1).unsqueeze( 1).expand(-1, 2, 2, 2)).view(-1,8,1)
    interpolated_feature = torch.sum(cell_feature * canonical_factor, axis=1)
    if len(xyz_shape) == 3:
        interpolated_feature = interpolated_feature.view(xyz.shape[0], xyz.shape[1], -1)
    return interpolated_feature

class SingleLayerTetra(nn.Module):
    def __init__(self,
                point_count,
                original_count,
                feature_dim):
        super().__init__()
        self.register_buffer("point_count", point_count)
        self.register_buffer("original_count", torch.tensor(original_count))
        self.register_buffer("extend_count", point_count - original_count)
        self.feature_dim = feature_dim
        # can be divided into two parts, 
        # the first `self.original_count` rows for original points,
        # the following `self.extend_count` rows for extend points.
        self.register_parameter(
            "field",
            nn.Parameter(
                torch.zeros((self.point_count, self.feature_dim), dtype=torch.float32)
            ),
        )
        self.register_buffer(
            "xyz",
            torch.zeros((self.point_count, 3),dtype=torch.float32)
        )
        # note: child_index < original_count for child in original field, child_index >= original_count for child in extend field; child_index == -1 for non existing child(should use interpolation); child_index[:,0] := -1
        self.register_buffer(
            "child_index",
            torch.zeros((self.point_count, 27), dtype=torch.int32)
        )
        self.register_buffer(
            "neighbor_index",
            torch.zeros((self.point_count, 7), dtype=torch.int32)
        )
        self.register_buffer(
            "parent_index",
            torch.zeros((self.point_count), dtype=torch.int32)
        )
        self.register_buffer(
            "importance",
            torch.zeros((self.point_count), dtype=torch.float32)
        )

    def search_next_layer(self, xyz, parent_cell, child_layer, mode="extend"):
        '''
        given a batched coordinates `xyz`, and a batched known cubic cells `parent_cell`, 
        the child_layer to be searched and a mode determining whether to use the extend field("original" or "extend").
        return a child cell, that is property(index in child_layer, feature value, xyz) of the eight vertices of the cell;
        if a vertex does not exist in the child_layer, set its feature value to be the interpolation value in the parent cell,
        its index to be nan or something.
        '''
        # parent_cell = {"vertices_id":cell_vertices_id, "feature":cell_feature, "xyz":cell_xyz}
        cell_vertices_id = parent_cell["vertices_id"]
        cell_feature = parent_cell["feature"]
        cell_xyz = parent_cell["xyz"]

        canonical = (xyz - cell_xyz[:,0,:]) / (cell_xyz[:,7,:] - cell_xyz[:,0,:])
        canonical_child_grid = torch.tensor([0,1/3,2/3,1], device=canonical)
        satisfied_x = (canonical_child_grid.unsqueeze(0).expand(canonical.shape[:-1]) <= canonical[:,:,0])
        satisfied_y = (canonical_child_grid.unsqueeze(0).expand(canonical.shape[:-1]) <= canonical[:,:,1])
        satisfied_z = (canonical_child_grid.unsqueeze(0).expand(canonical.shape[:-1]) <= canonical[:,:,2])
        child_cell_id_x = torch.sum(satisfied_x, axis=1) - 1
        child_cell_id_y = torch.sum(satisfied_y, axis=1) - 1
        child_cell_id_z = torch.sum(satisfied_z, axis=1) - 1

        # first level index
        parent_id = torch.ones_like(cell_vertices_id)
        parent_id[:,0] = cell_vertices_id[:,0]
        parent_id[:,[1,3,5,7]] = torch.where(child_cell_id_z==2, \
                                            cell_vertices_id[:,1].unsqueeze(-1), \
                                            parent_id[:,0].unsqueeze(-1))
        parent_id[:,[2,3,6,7]] = torch.where(child_cell_id_y==2, \
                                            cell_vertices_id[:,2].unsqueeze(-1), \
                                            parent_id[:,0])
        parent_id[:,[4,5,6,7]] = torch.where(child_cell_id_x==2, \
                                            cell_vertices_id[:,4].unsqueeze(-1), \
                                            parent_id[:,0].unsqueeze(-1))
        parent_id[:,[3,7]] = torch.where(torch.logical_and(child_cell_id_y==2, child_cell_id_z==2), \
                                        cell_vertices_id[:,3].unsqueeze(-1), \
                                        parent_id[:,3].unsqueeze(-1))
        parent_id[:,[5,7]] = torch.where(torch.logical_and(child_cell_id_x==2, child_cell_id_z==2), \
                                        cell_vertices_id[:,5].unsqueeze(-1), \
                                        parent_id[:,5].unsqueeze(-1))
        parent_id[:,[6,7]] = torch.where(torch.logical_and(child_cell_id_x==2, child_cell_id_y==2), \
                                        cell_vertices_id[:,6].unsqueeze(-1), \
                                        parent_id[:,6].unsqueeze(-1))
        parent_id[:,7] = torch.where(torch.logical_and(torch.logical_and(child_cell_id_x==2, \
                                                                        child_cell_id_x==2,  \
                                                                        child_cell_id_x==2)), \
                                    cell_vertices_id[:,7].unsqueeze(-1), \
                                    parent_id[:,7].unsqueeze(-1))

        # second level index
        child_id = torch.ones_like(cell_vertices_id)
        child_id_base = torch.stack((child_cell_id_x, child_cell_id_y, child_cell_id_z), axis=1).view(-1,1,3)
        child_id_offset = torch.zeros(2,2,2,3, dtype=torch.int32, device=child_id.device)
        child_id_offset[1,:,:,0] = 1
        child_id_offset[:,1,:,1] = 1
        child_id_offset[:,:,1,2] = 1
        child_id_3d = (child_id_base + child_id_offset.view(1,8,3)) % torch.tensor([[[3,3,3]]], dtype=torch.int32, device=cell_vertices_id.device)
        child_id = torch.sum( (child_id_3d)*torch.tensor([[[9,3,1]]], dtype=torch.int32, device=cell_vertices_id.device), axis=-1)

        # final level index
        child_cell_vertices_id = self.child_index[parent_id, child_id]
        child_cell_vertices_id = torch.where(parent_id == -1, -1, child_cell_vertices_id)

        # set return dict
        child_cell_canonical = (child_id_base + child_id_offset.view(1,8,3)) / 3
        child_cell_xyz = cell_xyz[:,0,:] + child_cell_canonical * (cell_xyz[:,7,:]-cell_xyz[:,0,:]).unsqueeze(1)
        child_cell_feature = child_layer.field[child_cell_vertices_id, :]
        # when get `child_cell_vertices_id == -1`, it means that the child cell is not set. (although indexing in the last layer would get the last element of the array instead of throwing error)
        child_cell_invalid_mask = (child_cell_vertices_id == -1)
        if mode == "original":
            child_cell_extend_mask = (child_cell_vertices_id >= child_layer.original_count)
            child_cell_invalid_mask = torch.logical_or(child_cell_invalid_mask, child_cell_extend_mask)
        #! the interpolation better involve other sampled points in this cell

        child_cell_interpolate_feature = interpolate_field_value(child_cell_xyz, parent_cell)
        child_cell_feature = torch.where(child_cell_invalid_mask, child_cell_interpolate_feature, child_cell_feature)
        return {"vertices_id":child_cell_vertices_id, "feature":child_cell_feature, "xyz":child_cell_xyz}

    def merge_extend_field(self):
        '''
        merge the extend field to the original field and empty the extend field;
        the extend field and the original field must be on the same layer.
        point number after merging should be determined according to distribution, for example, merge 13 out of 27
        '''
        #TODO: force the number of original points to be less then 1/4 of point_count
        pass

    def guide_next_layer(self, child_layer):
        '''
        guide the creation of extend field of a child_layer, after it is created or emptied.
        create children vertices for original_points as many as possible, 
        init feature values to be interpolation values, parent/neighbor/child index and xyz as they should be.
        #! remember to set child_index[:,0] = -1
        '''
        if child_extend_point_count > child_layer.point_count:
            #TODO: each original point should have different number of extend points
            #TODO: 26->8->4->1
            pass

    def organize_field_0(self, aabb):
        resolution = int(pow(self.point_count, 1/3)+0.1)
        
        min_x = aabb[0][0]
        min_y = aabb[0][1]
        min_z = aabb[0][2]
        max_x = aabb[1][0]
        max_y = aabb[1][1]
        max_z = aabb[1][2]
        axis_grid = torch.arange(resolution, device=self.field.device).unsqueeze(0).unsqueeze(0).expand((resolution, resolution, resolution))
        x = ( min_x + (max_x-min_x)/(resolution-1)*axis_grid ).transpose(2,0)
        y = ( min_y + (max_y-min_y)/(resolution-1)*axis_grid ).transpose(2,1)
        z = ( min_z + (max_z-min_z)/(resolution-1)*axis_grid )
        self.xyz.copy_(torch.stack((x,y,z), axis=-1).reshape(-1,3))

        index_grid = torch.arange(self.point_count, device=self.field.device).view(resolution, resolution, resolution)
        offset = index_grid[:2,:2,:2].reshape(8)[1:].unsqueeze(0).expand(resolution, resolution, resolution, 7)
        neighbor = index_grid.unsqueeze(-1).expand(-1, -1, -1, 7) + offset
        neighbor[-1,:,:,[3,4,5,6]] = -1
        neighbor[:,-1,:,[1,2,5,6]] = -1
        neighbor[:,:,-1,[0,2,4,6]] = -1
        self.neighbor_index.copy_(neighbor.view(-1,7))
        self.parent_index.copy_(-1*torch.ones_like(self.parent_index))

    def search_field_0(self, xyz):
        resolution = int(pow(self.point_count, 1/3)+0.1)

        xyz: Tensor[sample_num, 3]
        self.xyz: Tensor[resolution, resolution, resolution, 3]
        xyz_expand = xyz.unsqueeze(1).expand(-1, resolution, 3).view(-1, resolution, 3)
        self_xyz_reshape = self.xyz.view(resolution, resolution, resolution, 3)
        satisfied_x = (self_xyz_reshape[:,0,0,0].unsqueeze(0).expand(xyz_expand.shape[:-1]) <= xyz_expand[:,:,0])
        satisfied_y = (self_xyz_reshape[0,:,0,1].unsqueeze(0).expand(xyz_expand.shape[:-1]) <= xyz_expand[:,:,1])
        satisfied_z = (self_xyz_reshape[0,0,:,2].unsqueeze(0).expand(xyz_expand.shape[:-1]) <= xyz_expand[:,:,2])
        cell_id_x = torch.sum(satisfied_x, axis=1) - 1
        cell_id_y = torch.sum(satisfied_y, axis=1) - 1
        cell_id_z = torch.sum(satisfied_z, axis=1) - 1
        # cell_id = torch.stack((cell_id_x, cell_id_y, cell_id_z), axis=-1) - 1
        cell_id = resolution*(resolution*cell_id_x + cell_id_y) + cell_id_z

        neighbor_id = self.neighbor_index[cell_id, :]
        cell_vertices_id = torch.concatenate((cell_id.view(-1, 1), neighbor_id), axis=1)
        cell_feature = self.field[cell_vertices_id, :]
        cell_xyz = self.xyz[cell_vertices_id, :]
        return {"vertices_id":cell_vertices_id, "feature":cell_feature, "xyz":cell_xyz}

class MultiLayerTetra(nn.Module):
    def __init__(self,
                aabb,
                feature_dim = 32,
                xyz_resolution=torch.tensor([4,4,4]),
                max_layer_num=8,
                max_point_per_layer=1e5):
        super().__init__()
        # this list is hard to be reorganized, if using model.load, this list will not reorganize itself, but only the underlying parameters and buffers
        self.add_module("tetra_list", nn.ModuleList())
        self.register_buffer("aabb", aabb)
        self.register_buffer("feature_dim", torch.tensor(feature_dim))
        self.register_buffer("xyz_resolution", xyz_resolution)
        self.register_buffer("max_layer_num", torch.tensor(max_layer_num))
        self.register_buffer("max_point_per_layer", torch.tensor(max_point_per_layer))
        self.register_buffer("first_layer_point_num", torch.prod(xyz_resolution))
        assert 27*self.first_layer_point_num <= max_point_per_layer, "Variable `max_point_per_layer` is too small, it at least need to contain all points on the second layer"

        self.set_mode("original")

        self.tetra_list.append(
            SingleLayerTetra(self.first_layer_point_num, 0, feature_dim)
        )
        self.tetra_list[0].organize_field_0(aabb)
        self.refine_samples()

    def set_mode(self, mode):
        # mode can be "original" or "extend"
        self.mode = mode

    def refine_samples(self):
        for i in range(1, len(self.tetra_list)):
            self.tetra_list[i].merge_extend_field()
        l = len(self.tetra_list)
        if l < self.max_layer_num-1:
            point_count = min( pow(27, l)*self.first_layer_point_num, self.max_point_per_layer)
            self.tetra_list.append(
                SingleLayerTetra(point_count, point_count, self.feature_dim).to(self.aabb.device)
            )
            l += 1
        for i in range(l-1):
            self.tetra_list[i].guide_next_layer(self.tetra_list[i+1])

    def forward(self, xyz):
        cells = []
        cells.append(self.tetra_list[0].search_field_0(xyz))
        for i in range(len(self.tetra_list)-1):
            cells.append(self.tetra_list[i].search_next_layer(xyz, cells[-1], self.tetra_list[i+1], mode=self.mode))
        return interpolate_field_value(xyz, cells[-1])

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
        self.grid_resolution = torch.tensor([min_resolution, min_resolution, min_resolution])
        self.feature_dim = feature_dim

        self.geo_feat_dim = geo_feat_dim
        self.spatial_distortion = spatial_distortion

        self.direction_encoding = SHEncoding(
            levels=4,
            implementation=implementation,
        )

        self.multi_layer_tetra = MultiLayerTetra(aabb, feature_dim, self.grid_resolution, max_layer_num, max_point_per_layer)

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
