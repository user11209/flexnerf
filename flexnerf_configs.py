from __future__ import annotations

from nerfstudio.configs.base_config import ViewerConfig
from nerfstudio.data.datamanagers.base_datamanager import VanillaDataManagerConfig
from nerfstudio.data.dataparsers.blender_dataparser import BlenderDataParserConfig
from nerfstudio.data.dataparsers.dnerf_dataparser import DNeRFDataParserConfig
from nerfstudio.engine.optimizers import AdamOptimizerConfig
from nerfstudio.engine.schedulers import CosineDecaySchedulerConfig
from nerfstudio.engine.trainer import TrainerConfig
from nerfstudio.pipelines.base_pipeline import VanillaPipelineConfig
from nerfstudio.plugins.types import MethodSpecification

from .flex_nerf import FlexNeRFModelConfig

flex_nerf_method = TrainerConfig(
    method_name="flexnerf",
    steps_per_eval_batch=500,
    steps_per_save=2000,
    max_num_iterations=30000,
    mixed_precision=True,
    pipeline=VanillaPipelineConfig(
        datamanager=VanillaDataManagerConfig(
            dataparser=NerfstudioDataParserConfig(),
            train_num_rays_per_batch=4096,
            eval_num_rays_per_batch=4096,
            camera_optimizer=CameraOptimizerConfig(
                mode="SO3xR3",
                optimizer=AdamOptimizerConfig(lr=6e-4, eps=1e-8, weight_decay=1e-2),
                scheduler=ExponentialDecaySchedulerConfig(lr_final=6e-6, max_steps=200000),
            ),
        ),
        model=FlexNeRFModelConfig(eval_num_rays_per_chunk=1 << 15),
    ),
    optimizers={
        "proposal_networks": {
            "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-15),
            "scheduler": ExponentialDecaySchedulerConfig(lr_final=0.0001, max_steps=200000),
        },
        "fields": {
            "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-15),
            "scheduler": ExponentialDecaySchedulerConfig(lr_final=0.0001, max_steps=200000),
        },
    },
    viewer=ViewerConfig(num_rays_per_chunk=1 << 15),
    vis="viewer",
)

flex_nerf_dynamic_method = MethodSpecification(
    config=TrainerConfig(
        method_name="flex-nerf",
        steps_per_eval_batch=500,
        steps_per_save=2000,
        steps_per_eval_all_images=30000,
        max_num_iterations=30001,
        mixed_precision=True,
        pipeline=VanillaPipelineConfig(
            datamanager=VanillaDataManagerConfig(
                dataparser=DNeRFDataParserConfig(),
                train_num_rays_per_batch=4096,
                eval_num_rays_per_batch=4096,
                camera_res_scale_factor=0.5,  # DNeRF train on 400x400
            ),
            model=FlexNeRFModelConfig(
                eval_num_rays_per_chunk=1 << 15,
                grid_base_resolution=[128, 128, 128, 25],  # time-resolution should be half the time-steps
                grid_feature_dim=32,
                multiscale_res=[1, 2, 4],
                proposal_net_args_list=[
                    # time-resolution should be half the time-steps
                    {"num_output_coords": 8, "resolution": [128, 128, 128, 25]},
                    {"num_output_coords": 8, "resolution": [256, 256, 256, 25]},
                ],
                loss_coefficients={
                    "interlevel": 1.0,
                    "distortion": 0.01,
                    "plane_tv": 0.1,
                    "plane_tv_proposal_net": 0.0001,
                    "l1_time_planes": 0.001,
                    "l1_time_planes_proposal_net": 0.0001,
                    "time_smoothness": 0.1,
                    "time_smoothness_proposal_net": 0.001,
                },
            ),
        ),
        optimizers={
            # "proposal_networks": {
            #     "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-12),
            #     "scheduler": CosineDecaySchedulerConfig(warm_up_end=512, max_steps=30000),
            # },
            "tetrahedron_fields": {
                "optimizer": RAdamOptimizerConfig(lr=0.001),
                "scheduler": ExponentialDecaySchedulerConfig(
                    lr_final=0.0001,
                    max_steps=300_000,
                ),
            },
            "deformation_fields": {
                "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-12),
                "scheduler": CosineDecaySchedulerConfig(warm_up_end=512, max_steps=30000),
            },
        },
        viewer=ViewerConfig(num_rays_per_chunk=1 << 15),
        vis="viewer",
    ),
    description="K-Planes NeRF model for dynamic scenes"
)