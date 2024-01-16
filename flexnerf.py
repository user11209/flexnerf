# Copyright 2022 the Regents of the University of California, Nerfstudio Team and contributors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
NeRF implementation that combines many recent advancements.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Literal, Tuple, Type

import numpy as np
import torch
from torch.nn import Parameter
from torchmetrics.functional import structural_similarity_index_measure
from torchmetrics.image import PeakSignalNoiseRatio
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

from nerfstudio.cameras.rays import RayBundle, RaySamples
from nerfstudio.engine.callbacks import TrainingCallback, TrainingCallbackAttributes, TrainingCallbackLocation
from nerfstudio.field_components.field_heads import FieldHeadNames
from nerfstudio.field_components.spatial_distortions import SceneContraction
from nerfstudio.fields.density_fields import HashMLPDensityField
from nerfstudio.model_components.losses import (
    MSELoss,
    distortion_loss,
    interlevel_loss,
    orientation_loss,
    pred_normal_loss,
    scale_gradients_by_distance_squared,
)
from nerfstudio.model_components.ray_samplers import ProposalNetworkSampler, UniformSampler, UniformLinDispPiecewiseSampler
from nerfstudio.model_components.renderers import AccumulationRenderer, DepthRenderer, NormalsRenderer, RGBRenderer
from nerfstudio.model_components.scene_colliders import NearFarCollider
from nerfstudio.model_components.shaders import NormalsShader
from nerfstudio.models.base_model import Model, ModelConfig
from nerfstudio.utils import colormaps

from .flexnerf_field import FlexNerfField
from functools import partial

DEBUG = True
DEBUG_file = "/home/zhangjw/nerfstudio/external/flexnerf/log_file.txt"
DEBUG_global_i = 0

@dataclass
class FlexNeRFModelConfig(ModelConfig):
    """FlexNeRF Model Config"""

    _target: Type = field(default_factory=lambda: FlexNeRFModel)
    near_plane: float = 0.05
    """How far along the ray to start sampling."""
    far_plane: float = 2
    """How far along the ray to stop sampling."""
    background_color: Literal["random", "last_sample", "black", "white"] = "last_sample"
    """Whether to randomize the background color."""
    hidden_dim: int = 64
    """Dimension of hidden layers"""
    hidden_dim_color: int = 64
    """Dimension of hidden layers for color network"""
    hidden_dim_transient: int = 64
    """Dimension of hidden layers for transient network"""
    num_levels: int = 16
    """Number of levels of the hashmap for the base mlp."""
    base_res: int = 16
    """Resolution of the base grid for the hashgrid."""
    max_res: int = 2048
    """Maximum resolution of the hashmap for the base mlp."""
    log2_hashmap_size: int = 19
    """Size of the hashmap for the base mlp"""
    features_per_level: int = 2
    """How many hashgrid features per level"""
    num_proposal_samples_per_ray: Tuple[int, ...] = (256, 96)
    """Number of samples per ray for each proposal network."""
    num_nerf_samples_per_ray: int = 48
    """Number of samples per ray for the nerf network."""
    proposal_update_every: int = 5
    """Sample every n steps after the warmup"""
    proposal_warmup: int = 5000
    """Scales n from 1 to proposal_update_every over this many steps"""
    num_proposal_iterations: int = 2
    """Number of proposal network iterations."""
    use_same_proposal_network: bool = False
    """Arguments for the proposal density fields."""
    proposal_initial_sampler: Literal["piecewise", "uniform"] = "piecewise"
    """Initial sampler for the proposal network. Piecewise is preferred for unbounded scenes."""
    interlevel_loss_mult: float = 1.0
    """Proposal loss multiplier."""
    distortion_loss_mult: float = 0.002
    """Distortion loss multiplier."""
    orientation_loss_mult: float = 0.0001
    """Orientation loss multiplier on computed normals."""
    pred_normal_loss_mult: float = 0.001
    """Predicted normal loss multiplier."""
    use_proposal_weight_anneal: bool = True
    """Whether to use proposal weight annealing."""
    use_average_appearance_embedding: bool = True
    """Whether to use average appearance embedding or zeros for inference."""
    proposal_weights_anneal_slope: float = 10.0
    """Slope of the annealing function for the proposal weights."""
    proposal_weights_anneal_max_num_iters: int = 1000
    """Max num iterations for the annealing function."""
    use_single_jitter: bool = True
    """Whether use single jitter or not for the proposal networks."""
    predict_normals: bool = False
    """Whether to predict normals or not."""
    disable_scene_contraction: bool = False
    """Whether to disable scene contraction or not."""
    use_gradient_scaling: bool = False
    """Use gradient scaler where the gradients are lower for points closer to the camera."""
    implementation: Literal["tcnn", "torch"] = "torch"
    """Which implementation to use for the model."""
    appearance_embed_dim: int = 32
    """Dimension of the appearance embedding."""


class FlexNeRFModel(Model):
    """FlexNeRF model

    Args:
        config: FlexNeRF configuration to instantiate model
    """

    config: FlexNeRFModelConfig

    def populate_modules(self):
        """Set the fields and modules."""
        super().populate_modules()

        if self.config.disable_scene_contraction:
            scene_contraction = None
        else:
            scene_contraction = SceneContraction(order=float("inf"))

        # Fields
        self.field = FlexNerfField(
            self.scene_box.aabb,
            hidden_dim=self.config.hidden_dim,
            hidden_dim_color=self.config.hidden_dim_color,
            spatial_distortion=scene_contraction,
            implementation=self.config.implementation,
        )

        # Samplers
        self.sampler_uniform = UniformSampler(num_samples=self.config.num_nerf_samples_per_ray)

        # Collider
        self.collider = NearFarCollider(near_plane=self.config.near_plane, far_plane=self.config.far_plane)

        # renderers
        self.renderer_rgb = RGBRenderer(background_color=self.config.background_color)
        self.renderer_accumulation = AccumulationRenderer()
        self.renderer_depth = DepthRenderer()
        self.renderer_normals = NormalsRenderer()

        # losses
        self.rgb_loss = MSELoss()

        # metrics
        self.psnr = PeakSignalNoiseRatio(data_range=1.0)
        self.ssim = structural_similarity_index_measure
        self.lpips = LearnedPerceptualImagePatchSimilarity(normalize=True)

        # update field
        self.result = {'cell_id':None, 'extend_cell_id':None,
                      'weights_original':None, 'rgb_original':None, 
                      'weights_extend':None, 'rgb_extend':None, 
                      'pred_rgb':None, 'gt_rgb':None, 'density':None}

    def get_param_groups(self) -> Dict[str, List[Parameter]]:
        param_groups = {}
        param_groups["fields"] = list(self.field.parameters())
        return param_groups

    def get_training_callbacks(
        self, training_callback_attributes: TrainingCallbackAttributes
    ) -> List[TrainingCallback]:
        callbacks = []
        kwargs_dict = {"result": self.result}
        #? should be put into config
        importance_profile_start_step = 2000
        update_importance_callback = TrainingCallback(
                    where_to_run=[TrainingCallbackLocation.AFTER_TRAIN_ITERATION],
                    update_every_num_iters=1,
                    func=partial(self.field.multi_layer_tetra.update_importance_from_result, 
                                start_step=importance_profile_start_step),
                    kwargs=kwargs_dict,
                )
            )

        global_annealing_callback = TrainingCallback(
                    where_to_run=[TrainingCallbackLocation.AFTER_TRAIN_ITERATION],
                    update_every_num_iters=250,
                    func=partial(self.field.multi_layer_tetra.global_importance_annealing_callback,
                                start_step=importance_profile_start_step),
                )

        false_refine_callback = TrainingCallback(
                    where_to_run=[TrainingCallbackLocation.AFTER_TRAIN_ITERATION],
                    update_every_num_iters=50,
                    func=partial(self.field.multi_layer_tetra.refine_samples, update_original=False, 
                                start_step=importance_profile_start_step),
                )

        true_refine_callback = TrainingCallback(
                    where_to_run=[TrainingCallbackLocation.AFTER_TRAIN_ITERATION],
                    update_every_num_iters=1000,
                    func=partial(self.field.multi_layer_tetra.refine_samples, update_original=True,
                                start_step=importance_profile_start_step),
                )
                
        def update_step(step):
            self.field.multi_layer_tetra.step = step
        update_step_callback = TrainingCallback(
            where_to_run=[TrainingCallbackLocation.AFTER_TRAIN_ITERATION],
            update_every_num_iters=1,
            func=update_step,
        )
        
        callbacks.append(update_importance_callback)
        callbacks.append(global_annealing_callback)
        callbacks.append(false_refine_callback)
        callbacks.append(true_refine_callback)
        callbacks.append(update_step_callback)
        return callbacks

    def get_outputs(self, ray_bundle: RayBundle):
        ray_samples: RaySamples
        ray_samples = self.sampler_uniform(ray_bundle)
        field_outputs, extend_field_outputs = self.field.forward(ray_samples)
        if self.config.use_gradient_scaling:
            field_outputs = scale_gradients_by_distance_squared(field_outputs, ray_samples)
            extend_field_outputs = scale_gradients_by_distance_squared(extend_field_outputs, ray_samples)

        weights = ray_samples.get_weights(field_outputs[FieldHeadNames.DENSITY])

        rgb = self.renderer_rgb(rgb=field_outputs[FieldHeadNames.RGB], weights=weights, background_color="black")
        if DEBUG:
            if torch.any(torch.isnan(rgb)):
                nan_id = torch.nonzero(torch.any(torch.isnan(rgb), axis=-1)).view(-1)[0]
                with open(DEBUG_file, "a") as write_file:
                    write_file.write("look, rendering get an rgb of nan at id " + str(nan_id) + ".\n")
                    write_file.write("we need to list almost all possible info, for example: \n")
                    write_file.write("============== weights =============\n")
                    write_file.write(str(weights[nan_id]))
                    write_file.write("\n============== density =============\n")
                    write_file.write(str(field_outputs[FieldHeadNames.DENSITY][nan_id]))
                    write_file.write("\n============== rgb =================\n")
                    write_file.write(str(field_outputs[FieldHeadNames.RGB][nan_id]))
                    write_file.write("\n====================== Assertion Failed =====================")
                assert 0

        depth = self.renderer_depth(weights=weights, ray_samples=ray_samples)
        accumulation = self.renderer_accumulation(weights=weights)

        #? change the `True` on the next line to be something in `self.config`, about computing extend
        if True:
            weights_extend = ray_samples.get_weights(extend_field_outputs[FieldHeadNames.DENSITY])
            rgb_extend = self.renderer_rgb(rgb=extend_field_outputs[FieldHeadNames.RGB], weights=weights_extend, background_color="black")

        #? change the `True` on the next line to be something in `self.config`, about visualizing importance
        if True:
            @torch.inference_mode()
            def importance2density_n_rgb(importance, cell_id):
                # for importance
                importance_rgb_raw = ((torch.log10(importance + 1e-8) + 8) / 9).expand(-1,-1,3)
                # for cellis
                rgb_mid = cell_id.float()*12345.678
                rgb_raw = torch.cat([torch.sin(rgb_mid), torch.sin(1.11*rgb_mid), torch.sin(1.23*rgb_mid)], axis=-1)
                rgb_raw = (rgb_raw + 1)/2

                return importance_rgb_raw, rgb_raw
            
            importance_value = field_outputs[FieldHeadNames.IMPORTANCE]
            cell_id = field_outputs[FieldHeadNames.CELLID]
            extend_cell_id = extend_field_outputs[FieldHeadNames.CELLID]

            global DEBUG_global_i
            DEBUG_global_i += 1
            if DEBUG and DEBUG_global_i%100==0 and DEBUG_global_i>=2000:
                with open(DEBUG_file, "a") as write_file:
                    write_file.write("the maximum importance sampled is "+ str(torch.max(importance_value).item())+ ".\n")

            importance_rgb_raw, cellid_rgb_raw = importance2density_n_rgb(importance_value, cell_id)

        # update self.result for callbacks
        self.result["cell_id"] = cell_id
        self.result["extend_cell_id"] = extend_cell_id
        self.result["weights_original"] = weights
        self.result["rgb_original"] = field_outputs[FieldHeadNames.RGB]
        self.result["weights_extend"] = weights_extend
        self.result["rgb_extend"] = extend_field_outputs[FieldHeadNames.RGB]
        self.result["density"] = field_outputs[FieldHeadNames.DENSITY]

        with torch.inference_mode():
            slice_index = 65
            slice_density = field_outputs[FieldHeadNames.DENSITY].clone()
            slice_density[:, slice_index] = 10000
            slice_cellid_rgb_raw = field_outputs[FieldHeadNames.RGB].clone()
            slice_cellid_rgb_raw[:, slice_index, :] = cellid_rgb_raw[:, slice_index, :]
            weights_slice = ray_samples.get_weights(slice_density)
            weights_slice_clear = torch.zeros_like(weights_slice)
            weights_slice_clear[:, slice_index] = 1
            rgb_cellid_slice = self.renderer_rgb(rgb=slice_cellid_rgb_raw, weights=weights_slice)
            rgb_importance_slice_clear = self.renderer_rgb(rgb=importance_rgb_raw, weights=weights_slice_clear)
            rgb_cellid_slice_clear = self.renderer_rgb(rgb=slice_cellid_rgb_raw, weights=weights_slice_clear)
            rgb_cellid_onobj = self.renderer_rgb(rgb=cellid_rgb_raw, weights=weights, background_color="black")

        outputs = {
            "rgb": rgb,
            "accumulation": accumulation,
            "depth": depth,
            "rgb_extend": rgb_extend,
            "importance_slice_clear": rgb_importance_slice_clear,
            "cellid_slice_clear": rgb_cellid_slice_clear,
            "cellid_slice": rgb_cellid_slice,
            "cellid_onobj": rgb_cellid_onobj
        }

        return outputs

    def get_metrics_dict(self, outputs, batch):
        metrics_dict = {}
        gt_rgb = batch["image"].to(self.device)  # RGB or RGBA image
        gt_rgb = self.renderer_rgb.blend_background(gt_rgb)  # Blend if RGBA
        predicted_rgb = outputs["rgb"]
        metrics_dict["psnr"] = self.psnr(predicted_rgb, gt_rgb)

        return metrics_dict

    def get_loss_dict(self, outputs, batch, metrics_dict=None):
        loss_dict = {}
        image = batch["image"].to(self.device)
        pred_rgb, gt_rgb = self.renderer_rgb.blend_background_for_loss_computation(
            pred_image=outputs["rgb"],
            pred_accumulation=outputs["accumulation"],
            gt_image=image,
        )
        if DEBUG:
            pred_rgb_nan = torch.isnan(pred_rgb)
            gt_rgb_nan = torch.isnan(gt_rgb)
            if torch.any(pred_rgb_nan) or torch.any(gt_rgb_nan):
                if torch.any(pred_rgb_nan):
                    sample_id = torch.nonzero(torch.any(torch.isnan(pred_rgb), axis=-1)).view(-1)[0]
                if torch.any(gt_rgb_nan):
                    sample_id = torch.nonzero(torch.any(torch.isnan(gt_rgb), axis=-1)).view(-1)[0]
                with open(DEBUG_file, "a") as write_file:
                    write_file.write("====================== image =========================\n")
                    write_file.write(str(image[sample_id]))
                    write_file.write("\n====================== gt_rgb ========================\n")
                    write_file.write(str(gt_rgb[sample_id]))
                    write_file.write("\n====================== pred_rgb ======================\n")
                    write_file.write(str(pred_rgb[sample_id]))
                    write_file.write("\n====================== out_rgb =======================\n")
                    write_file.write(str(outputs["rgb"][sample_id]))
                    write_file.write("\n======================================================")
                assert 0

        extend_pred_rgb, extend_gt_rgb = self.renderer_rgb.blend_background_for_loss_computation(
            pred_image=outputs["rgb_extend"],
            pred_accumulation=outputs["accumulation"],
            gt_image=image,
        )

        # update self.result for callbacks
        self.result["pred_rgb"] = pred_rgb
        self.result["gt_rgb"] = gt_rgb

        loss_dict["rgb_loss"] = self.rgb_loss(gt_rgb, pred_rgb) + 0.1*self.rgb_loss(extend_gt_rgb, extend_pred_rgb)
        return loss_dict

    @torch.inference_mode()
    def get_image_metrics_and_images(
        self, outputs: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]
    ) -> Tuple[Dict[str, float], Dict[str, torch.Tensor]]:
        gt_rgb = batch["image"].to(self.device)
        predicted_rgb = outputs["rgb"]  # Blended with background (black if random background)
        gt_rgb = self.renderer_rgb.blend_background(gt_rgb)
        acc = colormaps.apply_colormap(outputs["accumulation"])
        importance_slice_clear = outputs["importance_slice_clear"]
        cellid_slice = outputs["cellid_slice"]
        cellid_slice_clear = outputs["cellid_slice_clear"]
        cellid_onobj = outputs["cellid_onobj"]
        depth = colormaps.apply_depth_colormap(
            outputs["depth"],
            accumulation=outputs["accumulation"],
        )

        combined_rgb = torch.cat([gt_rgb, predicted_rgb], dim=1)
        combined_acc = torch.cat([acc], dim=1)
        combined_depth = torch.cat([depth], dim=1)

        # Switch images from [H, W, C] to [1, C, H, W] for metrics computations
        gt_rgb = torch.moveaxis(gt_rgb, -1, 0)[None, ...]
        predicted_rgb = torch.moveaxis(predicted_rgb, -1, 0)[None, ...]

        psnr = self.psnr(gt_rgb, predicted_rgb)
        ssim = self.ssim(gt_rgb, predicted_rgb)
        lpips = self.lpips(gt_rgb, predicted_rgb)

        # all of these metrics will be logged as scalars
        metrics_dict = {"psnr": float(psnr.item()), "ssim": float(ssim)}  # type: ignore
        metrics_dict["lpips"] = float(lpips)

        images_dict = {"img": combined_rgb, "accumulation": combined_acc, "depth": combined_depth, 
                        "importance_slice_clear": importance_slice_clear, "cellid_slice": cellid_slice, "cellid_slice_clear": cellid_slice_clear,
                        "cellid_onobj": cellid_onobj}

        return metrics_dict, images_dict
