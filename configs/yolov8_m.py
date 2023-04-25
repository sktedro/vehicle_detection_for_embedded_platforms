import os
import sys
from copy import deepcopy

# No way to find out the path of this file since mmdetection does shady things
# with it, so we need an absolute filepath of the repository
repo_path = "/home/xskalo01/bp/proj/"
# repo_path = "/home/tedro/Desktop/d_projekty/bp/proj/"
sys.path.append(repo_path)
import paths
import dataset.common as common

sys.path.append(os.path.join(repo_path, "configs"))
from settings import num_gpus, img_scale, max_epochs, warmup_epochs, val_interval, save_epoch_intervals, base_lr


# Import custom modules - Custom CutOut
custom_imports = dict(imports=["custom_modules"], allow_failed_imports=False)

work_dir = paths.working_dirpath

default_scope = 'mmyolo'
deepen_factor = 0.67
widen_factor = 0.75

try:
    load_from = paths.get_last_checkpoint_filepath(work_dir)
    resume = True
except Exception as e:
    print(f"Not resuming training because checkpoint was not found: {str(e)}. Trying a pre-trained model")
    if paths.model_checkpoint_filepath:
        print(f"Pre-trained model found: {paths.model_checkpoint_filepath}")
        load_from = paths.model_checkpoint_filepath
    resume = False

max_keep_ckpts = 100

pre_trained_model_batch_size_per_gpu = 16 # 16 for YOLOv8

# Batch size (default 8)
# train_batch_size_per_gpu = 11 # YOLOv8-m, P52. 12 -> Cuda out of memory
# train_batch_size_per_gpu = 24 # YOLOv8-n, P52. 27 -> Cuda out of memory
train_batch_size_per_gpu = 46 # YOLOv8-m, Sophie with 640x384. 48 -> Cuda out of memory after tens of epochs
# train_batch_size_per_gpu = 26 # YOLOv8-m, Sophie with 640x640. 30 -> Cuda out of memory

# Workers per gpu (default 4)
# Tested 8, 12 and 16 on P52 and higher numbers actually made the training (ETA) longer
# With 12, ETA was about 10% longer than at default. Using 2, speed is slightly improved (~2%)
# train_num_workers = 6 # Doesn't seem to be better than 1. It even seems to be a bit slower (slightly lower CPU and GPU usage) - at least for YOLOv8-m
# train_num_workers = 1
train_num_workers = 8 # On Sophie (internal GPU), more workers is better

val_batch_size_per_gpu = 32
val_num_workers = 16

test_batch_size_per_gpu = 32
test_num_workers = 16

lr_factor = 0.01

metainfo = dict(
    classes = tuple(common.classes_ids.keys())
)
num_classes = len(metainfo["classes"])

file_client_args = dict(backend='disk')

min_gt_bbox_wh = (8, 8) # Default YOLOv8 is 1*1, but I imagine 8*8 being better
pad_val = 114

# TODO more (or more aggressive) augmentations since mAP decreases when training?
# Maybe random stretch along x or y axis?
train_pipeline = [
    dict(type='LoadImageFromFile',
        file_client_args=file_client_args),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='mmdet.Resize',
        scale=img_scale, # Shouldn't matter at all here
        keep_ratio=True),
    dict(type='mmdet.Pad',
        pad_to_square=False,
        size=img_scale, # width * height...
        pad_val=dict(img=(pad_val, pad_val, pad_val))),
    dict(type='YOLOv5RandomAffine',
        # min_bbox_size=8, # No need. Done in FilterAnnotations
        scaling_ratio_range=None, # Needs to be adjusted per dataset later below
        max_translate_ratio=0.05,
        max_rotate_degree=5,
        max_shear_degree=3,
        border_val=(pad_val, pad_val, pad_val)),
    dict(type="mmdet.CustomCutOut",
        prob=0.05,
        cutout_area=(0.05, 0.35),
        random_pixels=True),
    dict(type='mmdet.CutOut',
        n_holes=None, # Closed interval
        cutout_shape=None, # Patch size in px
        fill_in=(pad_val, pad_val, pad_val)),
    dict(type='mmdet.Albu', # As in YOLOv8 default config
        transforms=[
            dict(type='Blur', p=0.01),
            dict(type='MedianBlur', p=0.01),
            dict(type='ToGray', p=0.01),
            dict(type='CLAHE', p=0.01)],
        bbox_params=dict(
            type='BboxParams',
            format='pascal_voc',
            label_fields=['gt_bboxes_labels', 'gt_ignore_flags']),
        keymap={
            'img': 'image',
            'gt_bboxes': 'bboxes'
        }),
    dict(type='YOLOv5HSVRandomAug'),
    dict(type='mmdet.RandomFlip',
        prob=0.5),
    dict(type="mmdet.PhotoMetricDistortion"),
    dict(type='mmdet.FilterAnnotations',
        min_gt_bbox_wh=min_gt_bbox_wh,
        keep_empty=False), # Give it some negative images (if keep_empty=True, it returns None if there are no bboxes...)
    dict(type='mmdet.PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'flip', 'flip_direction')),
]

# Stage 2: no cutout and different affine transforms
train_pipeline_stage2 = deepcopy(train_pipeline)
assert train_pipeline_stage2[4]["type"] == "YOLOv5RandomAffine", "Assertion 844"
train_pipeline_stage2[4] = dict(
        type='YOLOv5RandomAffine',
        max_rotate_degree=0,
        max_shear_degree=0,
        scaling_ratio_range=(0.9, 1.2),
        max_aspect_ratio=100,
        border_val=(pad_val, pad_val, pad_val))
assert train_pipeline_stage2[5]["type"] == "mmdet.CustomCutOut", "Assertion 122"
del train_pipeline_stage2[5]
assert train_pipeline_stage2[5]["type"] == "mmdet.CutOut", "Assertion 324"
del train_pipeline_stage2[5]

train_datasets_repeats = {
    "mio-tcd"     : 1,
    "aau"         : 3, # There are some misannotations so don't make it too frequent
    "ndis"        : 25,
    "mtid"        : 5, # It's a video, so already a lot repeats, but it's a great dataset
    "visdrone_det": 4, # Good dataset, but not very important in this project
    "detrac"      : 1,
}

train_datasets_scaling_ratios = {
    "mio-tcd"     : (1.0, 1.1),
    "aau"         : (0.9, 1.1),
    "ndis"        : (0.9, 1.5),
    "mtid"        : (0.9, 2),
    "visdrone_det": (1.5, 2.5),
    "detrac"      : (0.8, 1),
}

# Individual cutout: [Number of holes (closed interval), (patch size in pixels)]
train_dataset_cutout_vals = {
    "mio-tcd"     : [ 4, (26, 26)],
    "aau"         : [ 8, (10, 10)],
    "ndis"        : [12, (20, 20)],
    "mtid"        : [12, (10, 10)],
    "visdrone_det": [20, ( 8,  8)],
    "detrac"      : [ 6, (22, 22)],
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
            metainfo = metainfo,
            type = "YOLOv5CocoDataset",
            ann_file = os.path.join(paths.datasets_dirpath,
                                    common.datasets[dataset_name]["path"],
                                    common.gt_filenames["train"]),
            data_prefix = dict(img=os.path.join(
                paths.datasets_dirpath,
                common.datasets[dataset_name]["path"])),
            data_root = os.path.join(paths.datasets_dirpath,
                                     common.datasets[dataset_name]["path"]),
            filter_cfg = dict(filter_empty_gt=False, min_size=32),
            pipeline = deepcopy(train_pipeline)))

    # Set RandomAffine scaling range individually for each dataset
    assert ds["dataset"]["pipeline"][4]["type"] == "YOLOv5RandomAffine", "Assertion 94"
    ds["dataset"]["pipeline"][4]["scaling_ratio_range"] = train_datasets_scaling_ratios[dataset_name]

    # Set CutOut number of holes and cutout shape (size) individually 
    assert ds["dataset"]["pipeline"][6]["type"] == "mmdet.CutOut", "Assertion 81"
    ds["dataset"]["pipeline"][6]["n_holes"] = train_dataset_cutout_vals[dataset_name][0]
    ds["dataset"]["pipeline"][6]["cutout_shape"] = train_dataset_cutout_vals[dataset_name][1]

    train_dataset["datasets"].append(ds)
    del ds # Delete it so it's not in the final config
del dataset_name # Delete it so it's not in the final config

train_dataloader = dict(
    batch_size = train_batch_size_per_gpu,
    num_workers = train_num_workers,

    persistent_workers = True,
    pin_memory=True,
    sampler=dict(type="DefaultSampler", shuffle=True),
    collate_fn=dict(type='yolov5_collate'),

    dataset = train_dataset
)
del train_dataset # Delete it so it's not in the final config

val_pipeline = [
    dict(type='LoadImageFromFile',
        file_client_args=file_client_args),
    dict(type='LoadAnnotations',
        with_bbox=True,
        _scope_='mmdet'),
    dict(type='YOLOv5KeepRatioResize', scale=img_scale),
    dict(type='LetterResize',
        scale=img_scale, # This takes w*h
        allow_scale_up=False,
        pad_val=dict(img=pad_val)),

    dict(type='mmdet.PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor', 'pad_param'))
]
# For some reason, test pipeline needs to be different...
test_pipeline = [
    dict(type='LoadImageFromFile',
        file_client_args=file_client_args),
    dict(type='LetterResize',
        scale=img_scale, # This takes w*h
        allow_scale_up=False,
        pad_val=dict(img=pad_val)),
    dict(type='LoadAnnotations',
        with_bbox=True,
        _scope_='mmdet'),
    dict(type='mmdet.PackDetInputs',
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
        metainfo = metainfo,
        type = "YOLOv5CocoDataset",
        data_root = paths.datasets_dirpath,
        ann_file = os.path.basename(common.gt_combined_filenames["val"]),
        data_prefix = dict(img=""),
        test_mode = True,
        pipeline = val_pipeline
    )
)
test_dataloader = dict(
    batch_size = test_batch_size_per_gpu,
    num_workers = test_num_workers,
    persistent_workers = True,
    drop_last = False,
    sampler = dict(type="DefaultSampler", shuffle=False),
    dataset = dict(
        metainfo = metainfo,
        type = "YOLOv5CocoDataset",
        data_root = paths.datasets_dirpath,
        ann_file = os.path.basename(common.gt_combined_filenames["test"]),
        data_prefix = dict(img=""),
        test_mode = True,
        pipeline = test_pipeline
    )
)

val_evaluator = dict(
    type='mmdet.CocoMetric',
    proposal_nums=(100, 1, 10),
    ann_file=os.path.join(paths.datasets_dirpath, common.gt_combined_filenames["val"]),
    metric='bbox')

test_evaluator = dict(
    type='mmdet.CocoMetric',
    proposal_nums=(100, 1, 10),
    ann_file=os.path.join(paths.datasets_dirpath, common.gt_combined_filenames["test"]),
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

# This doesn't seem to be necessary. The batch_size_per_gpu key in
# optim_wrapper.optimizer should do the trick (at least with MMYOLO)
#  auto_scale_lr = dict(
    #  enable=True,
    #  base_batch_size=train_batch_size_per_gpu,
#  )

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
        warmup_epochs = warmup_epochs),
    checkpoint=dict(
        type = 'CheckpointHook',
        interval = save_epoch_intervals,
        save_best = 'auto',
        max_keep_ckpts=max_keep_ckpts),
    sampler_seed = dict(type='DistSamplerSeedHook'),
    visualization = dict(type='mmdet.DetVisualizationHook'))

custom_hooks = [
    dict(
        type='EMAHook',
        ema_type='ExpMomentumEMA',
        momentum=0.0001,
        update_buffers=True,
        strict_load=False,
        priority=49),
    dict(
        type='mmdet.PipelineSwitchHook',
        switch_epoch=max_epochs - min(10, round(max_epochs / 10)), # when training for less than 100 epochs, have less stage2 epochs
        switch_pipeline=train_pipeline_stage2)
]

train_cfg = dict(
    type='EpochBasedTrainLoop',
    max_epochs=max_epochs,
    val_interval=val_interval,
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
    name='visualizer',
    line_width=2,
    alpha=0.9)

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
            featmap_strides=strides),
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
