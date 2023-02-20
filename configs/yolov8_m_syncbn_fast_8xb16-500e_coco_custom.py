import os
from copy import deepcopy

import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', "process_dataset"))
import paths
import process_dataset.common as common


default_scope = 'mmyolo'
deepen_factor = 0.67
widen_factor = 0.75

if paths.last_checkpoint_filepath:
    load_from = paths.last_checkpoint_filepath
    resume = True
else:
    load_from = paths.model_checkpoint_filepath

base_lr = 0.00125 # 0.00125 per gpu
lr_factor = 0.01
max_epochs = 500
save_epoch_intervals = 1
max_keep_ckpts = 100

# Batch size (default 8)
# train_batch_size_per_gpu = 11 # YOLOv8-m, P52. 12 -> Cuda out of memory
# train_batch_size_per_gpu = 24 # YOLOv8-n, P52. 27 -> Cuda out of memory
train_batch_size_per_gpu = 26 # YOLOv8-m, Sophie. 30 -> Cuda out of memory

# Workers per gpu (default 4)
# Tested 8, 12 and 16 on P52 and higher numbers actually made the training (ETA) longer
# With 12, ETA was about 10% longer than at default. Using 2, speed is slightly improved (~2%)
# train_num_workers = 6 # Doesn't seem to be better than 1. It even seems to be a bit slower (slightly lower CPU and GPU usage) - at least for YOLOv8-m
# train_num_workers = 1
train_num_workers = 16 # On Sophie (internal GPU), more workers is better

val_batch_size_per_gpu = 1
val_num_workers = 2

test_batch_size_per_gpu = 1
test_num_workers = 2

img_scale = (640, 360) # height, width
num_classes = 8
metainfo = dict(
    classes = tuple(common.classes_ids.keys())
)

work_dir = paths.working_dirpath
data_root = common.datasets_dirpath

file_client_args = dict(backend='disk')

# TODO + some other augs from yolov8?
train_pipeline = [
    dict(type='LoadImageFromFile',
        file_client_args=dict(backend='disk')),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='mmdet.Resize',
        scale=img_scale,
        keep_ratio=True),
    dict(type='mmdet.Pad',
        pad_to_square=True,
        pad_val=dict(img=(114.0, 114.0, 114.0))),
    dict(type='YOLOv5RandomAffine',
        # min_bbox_size=8, # No need. Done in FilterAnnotations
        scaling_ratio_range=(0, 0), # Needs to be adjusted per dataset later below
        max_rotate_degree=10,
        max_shear_degree=5),
    dict(type='YOLOv5HSVRandomAug'),
    dict(type='mmdet.RandomFlip',
         prob=0.5),
    dict(type="mmdet.PhotoMetricDistortion"),
    dict(type='mmdet.FilterAnnotations',
        # min_gt_bbox_wh=(8, 8), # Should be okay, I think 16x16 causes small objects (even 64x64) to be undetected
        min_gt_bbox_wh=(1, 1), # But YOLOX originally just uses 1x1, so let's try
        keep_empty=False),
    dict(type='mmdet.PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'flip', 'flip_direction')),
]

train_datasets_scaling_ratios = {
    "mio-tcd"     : (0.7, 1.1),
    "aau"         : (0.8, 1.1),
    "ndis"        : (0.9, 3),
    "mtid"        : (0.9, 2),
    "visdrone_det": (1.5, 3),
    "detrac"      : (0.8, 1.2)
}

train_datasets_repeats = {
    "mio-tcd"     : 1,
    "aau"         : 3, # There are some misannotations so don't make it too frequent
    "ndis"        : 25,
    "mtid"        : 6, # It's a video, so already a lot repeats, but it's a great dataset
    "visdrone_det": 4, # Good dataset, but not very important in this project
    "detrac"      : 2
}

train_dataset = dict(
    type = "ConcatDataset",
    datasets = []
)
for dataset_name in list(common.datasets.keys()):
    ds = dict(
        type = "RepeatDataset",
        times = train_datasets_repeats[dataset_name],
        dataset = dict(
            type = "YOLOv5CocoDataset",
            ann_file = os.path.join(common.datasets_dirpath, common.datasets[dataset_name]["path"], common.gt_filename),
            data_prefix = dict(img=data_root),
            data_root = data_root,
            filter_cfg = dict(filter_empty_gt=False, min_size=32),
            pipeline = deepcopy(train_pipeline),
        )
    )

    # Set RandomAffine scaling range individually for each dataset
    assert ds["dataset"]["pipeline"][4]["type"] == "YOLOv5RandomAffine"
    ds["dataset"]["pipeline"][4]["scaling_ratio_range"] = train_datasets_scaling_ratios[dataset_name]

    train_dataset["datasets"].append(ds)

train_dataloader = dict(
    batch_size = train_batch_size_per_gpu,

    num_workers = train_num_workers,

    persistent_workers = True,
    pin_memory=True,
    sampler=dict(type="DefaultSampler", shuffle=True),
    collate_fn=dict(type='yolov5_collate'),

    # TODO restore this and use class balanced dataset (was getting an exception when used)
    # "The dataset needs to instantiate self.get_cat_ids() to support ClassBalancedDataset."
    # So if I have ConcatDataset in ClassBalancedDataset, the ConcatDataset must have get_cat_ids()
    # dataset = dict(
    #     type = 'ClassBalancedDataset',
    #     # oversample_thr = 1e-3, # Default
    #     oversample_thr = 0.1, # Seems good
    #     dataset = train_dataset
    # ),

    # This works (omitting class balanced dataset)
    dataset = train_dataset
)

test_pipeline = [
    dict(type='LoadImageFromFile', file_client_args=file_client_args),
    dict(type='YOLOv5KeepRatioResize', scale=img_scale),
    dict(
        type='LetterResize',
        scale=img_scale,
        allow_scale_up=False,
        pad_val=dict(img=114)),
    dict(type='LoadAnnotations', with_bbox=True, _scope_='mmdet'),
    dict(
        type='mmdet.PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor', 'pad_param'))
]

val_dataloader = dict(
    batch_size = val_batch_size_per_gpu,
    num_workers = val_num_workers,
    persistent_workers = True,
    drop_last = False,
    sampler = dict(type="DefaultSampler", shuffle=False),
    dataset = dict(
        type = "YOLOv5CocoDataset",
        data_root = data_root,
        ann_file = os.path.basename(common.dataset_val_filepath),
        data_prefix = dict(img=""),
        test_mode = True,
        pipeline = test_pipeline,
    )
)

test_dataloader = dict(
    batch_size = test_batch_size_per_gpu,
    num_workers = test_num_workers,
    persistent_workers = True,
    drop_last = False,
    sampler = dict(type="DefaultSampler", shuffle=False),
    dataset = dict(
        type = "YOLOv5CocoDataset",
        data_root = data_root,
        ann_file = os.path.basename(common.dataset_test_filepath),
        data_prefix = dict(img=""),
        test_mode = True,
        pipeline = test_pipeline
    )
)

val_evaluator = dict(
    type='mmdet.CocoMetric',
    proposal_nums=(100, 1, 10),
    ann_file=common.dataset_val_filepath,
    metric='bbox')
test_evaluator = dict(
    type='mmdet.CocoMetric',
    proposal_nums=(100, 1, 10),
    ann_file=common.dataset_test_filepath,
    metric='bbox')

optim_wrapper = dict(
    type='OptimWrapper',
    clip_grad=dict(max_norm=10.0),
    optimizer=dict(
        type='SGD',
        lr=base_lr,
        momentum=0.937,
        weight_decay=0.0005,
        nesterov=True,
        batch_size_per_gpu=train_batch_size_per_gpu),
    constructor='YOLOv5OptimizerConstructor')

default_hooks = dict(
    timer = dict(type='IterTimerHook'),
    logger = dict(
        type='LoggerHook',
        interval=50),
    param_scheduler=dict(
        type = 'YOLOv5ParamSchedulerHook',
        scheduler_type = 'linear',
        lr_factor = lr_factor,
        max_epochs = max_epochs,
        warmup_epochs = 3),
    checkpoint=dict(
        type = 'CheckpointHook',
        interval = save_epoch_intervals,
        save_best = 'auto',
        max_keep_ckpts=max_keep_ckpts),
    sampler_seed = dict(type='DistSamplerSeedHook'),
    visualization = dict(type='mmdet.DetVisualizationHook'))

train_cfg = dict(
    type='EpochBasedTrainLoop',
    max_epochs=max_epochs,
    val_interval=save_epoch_intervals,
    dynamic_intervals=[(max_epochs - 10, 1)])
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

log_level = 'INFO'
log_processor = dict(type='LogProcessor', window_size=50, by_epoch=True)

visualizer = dict(
    type='mmdet.DetLocalVisualizer',
    vis_backends=[
        dict(type='LocalVisBackend'),
        dict(type='TensorboardVisBackend')
    ],
    name='visualizer')

env_cfg = dict(
    cudnn_benchmark=True,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'),
)

last_stage_out_channels = 768
strides = [8, 16, 32]
model = dict(
    type='YOLODetector',
    data_preprocessor=dict(
        type='YOLOv5DetDataPreprocessor',
        mean=[0., 0., 0.],
        std=[255., 255., 255.],
        bgr_to_rgb=True),
    backbone=dict(
        type='YOLOv8CSPDarknet',
        arch='P5',
        last_stage_out_channels=last_stage_out_channels,
        deepen_factor=deepen_factor,
        widen_factor=widen_factor,
        norm_cfg=dict(type='BN', momentum=0.03, eps=0.001),
        act_cfg=dict(type='SiLU', inplace=True)),
    neck=dict(
        type='YOLOv8PAFPN',
        deepen_factor=deepen_factor,
        widen_factor=widen_factor,
        in_channels=[256, 512, last_stage_out_channels],
        out_channels=[256, 512, last_stage_out_channels],
        num_csp_blocks=3,
        norm_cfg=dict(type='BN', momentum=0.03, eps=0.001),
        act_cfg=dict(type='SiLU', inplace=True)),
    bbox_head=dict(
        type='YOLOv8Head',
        head_module=dict(
            type='YOLOv8HeadModule',
            num_classes=num_classes,
            in_channels=[256, 512, last_stage_out_channels],
            widen_factor=widen_factor,
            reg_max=16,
            norm_cfg=dict(type='BN', momentum=0.03, eps=0.001),
            act_cfg=dict(type='SiLU', inplace=True),
            featmap_strides=[8, 16, 32]),
        prior_generator=dict(
            type='mmdet.MlvlPointGenerator', offset=0.5, strides=strides),
        bbox_coder=dict(type='DistancePointBBoxCoder'),
        loss_cls=dict(
            type='mmdet.CrossEntropyLoss',
            use_sigmoid=True,
            reduction='none',
            loss_weight=0.5),
        loss_bbox=dict(
            type='IoULoss',
            iou_mode='ciou',
            bbox_format='xyxy',
            reduction='sum',
            loss_weight=7.5,
            return_iou=False),
        # Since the dfloss is implemented differently in the official
        # and mmdet, we're going to divide loss_weight by 4.
        loss_dfl=dict(
            type='mmdet.DistributionFocalLoss',
            reduction='mean',
            loss_weight=1.5 / 4)),
    train_cfg=dict(
        assigner=dict(
            type='BatchTaskAlignedAssigner',
            num_classes=num_classes,
            use_ciou=True,
            topk=10,
            alpha=0.5,
            beta=6.0,
            eps=1e-9)),
    test_cfg=dict(
        multi_label=True,
        nms_pre=30000,
        score_thr=0.001,
        nms=dict(type='nms', iou_threshold=0.7),
        max_per_img=300))
