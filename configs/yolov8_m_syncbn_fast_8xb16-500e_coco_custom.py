import os
from copy import deepcopy

import sys, os
sys.path.append(os.path.join(os.getcwd(), '..'))
sys.path.append(os.path.join(os.getcwd(), '..', "process_dataset"))
import paths
import dataset.common as common


default_scope = 'mmyolo'
deepen_factor = 0.67
widen_factor = 0.75

if paths.last_checkpoint_filepath:
    load_from = paths.last_checkpoint_filepath
    resume = True
else:
    load_from = paths.model_checkpoint_filepath
    resume = False

num_gpus = 4

max_epochs = 300 # 500 # TODO
warmup_epochs = 3 # 3 # TODO
val_interval = 5 # 10 # TODO
save_epoch_intervals = 1
max_keep_ckpts = 100

pre_trained_model_batch_size_per_gpu = 16 # 16 for YOLOv8

# Batch size (default 8)
# train_batch_size_per_gpu = 11 # YOLOv8-m, P52. 12 -> Cuda out of memory
# train_batch_size_per_gpu = 24 # YOLOv8-n, P52. 27 -> Cuda out of memory
train_batch_size_per_gpu = 48 # YOLOv8-m, Sophie with 640x384. 52 -> Cuda out of memory
# train_batch_size_per_gpu = 26 # YOLOv8-m, Sophie with 640x640. 30 -> Cuda out of memory

# Workers per gpu (default 4)
# Tested 8, 12 and 16 on P52 and higher numbers actually made the training (ETA) longer
# With 12, ETA was about 10% longer than at default. Using 2, speed is slightly improved (~2%)
# train_num_workers = 6 # Doesn't seem to be better than 1. It even seems to be a bit slower (slightly lower CPU and GPU usage) - at least for YOLOv8-m
# train_num_workers = 1
train_num_workers = 4 # On Sophie (internal GPU), more workers is better

val_batch_size_per_gpu = 1
val_num_workers = 4

test_batch_size_per_gpu = 1
test_num_workers = 4

# Learning rate  = 0.00125 per gpu, linear to batch size (https://stackoverflow.com/questions/53033556/how-should-the-learning-rate-change-as-the-batch-size-change)
# Per gpu, because mmengine anyways says when training: LR is set based on batch size of [batch_size*num_gpus] and the current batch size is [batch_size]. Scaling the original LR by [1/num_gpus].
# base_lr = 0.00125 * num_gpus * (train_batch_size_per_gpu / pre_trained_model_batch_size_per_gpu)
base_lr = 0.00125 # Tried different LRs on 4 GPUs, 48 batch size, and 0.002 was worse and 0.0005 was worse.. So scaling doesn't seem to be needed... TODO try on single gpu with lower batch size?
lr_factor = 0.01

# When changing the image scale, don't forget to change batch size
# img_scale = (640, 640) # height, width; default
img_scale = (384, 640) # height, width; need to be multiples of 32

metainfo = dict(
    classes = tuple(common.classes_ids.keys())
)
num_classes = len(metainfo["classes"])

work_dir = paths.working_dirpath
data_root = paths.datasets_dirpath
file_client_args = dict(backend='disk')

min_gt_bbox_wh = (8, 8) # Default YOLO is 1*1, but I imagine 8*8 being better
pad_val = 114

train_pipeline = [
    dict(type='LoadImageFromFile',
        file_client_args=file_client_args),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='mmdet.Resize',
        scale=img_scale, # h*w I guess. Shouldn't matter at all here
        keep_ratio=True),
    dict(type='mmdet.Pad',
        pad_to_square=False,
        size=img_scale[::-1], # width * height...
        pad_val=dict(img=(pad_val, pad_val, pad_val))),
    dict(type='YOLOv5RandomAffine',
        # min_bbox_size=8, # No need. Done in FilterAnnotations
        scaling_ratio_range=None, # Needs to be adjusted per dataset later below
        max_rotate_degree=5,
        max_shear_degree=5),
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
        keep_empty=False),
    dict(type='mmdet.PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'flip', 'flip_direction')),
]

train_datasets_repeats = {
    "mio-tcd"     : 1,
    "aau"         : 3, # There are some misannotations so don't make it too frequent
    "ndis"        : 25,
    "mtid"        : 6, # It's a video, so already a lot repeats, but it's a great dataset
    "visdrone_det": 4, # Good dataset, but not very important in this project
    "detrac"      : 2,
}

train_datasets_scaling_ratios = {
    "mio-tcd"     : (0.9, 1.1),
    "aau"         : (0.9, 1.1),
    "ndis"        : (0.9, 2.5),
    "mtid"        : (0.9, 2),
    "visdrone_det": (1.5, 3),
    "detrac"      : (0.8, 1.2),
}

# Individual cutout: [Number of holes (closed interval), (patch size in pixels)]
train_dataset_cutout_vals = {
    "mio-tcd"     : [ 4, (32, 32)],
    "aau"         : [ 8, (12, 12)],
    "ndis"        : [12, (24, 24)],
    "mtid"        : [12, (12, 12)],
    "visdrone_det": [20, ( 8,  8)],
    "detrac"      : [ 6, (24, 24)],
}

# ConcatDataset -> RepeatDataset -> YOLOv5CocoDataset
# TODO Use class balanced dataset? (I was getting an exception when used)
# "The dataset needs to instantiate self.get_cat_ids() to support ClassBalancedDataset."
# So if I have ConcatDataset in ClassBalancedDataset, the ConcatDataset must have get_cat_ids()
# dataset = dict(
#     type = 'ClassBalancedDataset',
#     # oversample_thr = 1e-3, # Default
#     oversample_thr = 0.1, # Seems good
#     dataset = ...Concatdataset...
# ),
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
            ann_file = os.path.join(paths.datasets_dirpath, common.datasets[dataset_name]["path"], common.gt_filenames["train"]),
            data_prefix = dict(img=data_root),
            data_root = data_root,
            filter_cfg = dict(filter_empty_gt=False, min_size=32),
            pipeline = deepcopy(train_pipeline)))

    # Set RandomAffine scaling range individually for each dataset
    assert ds["dataset"]["pipeline"][4]["type"] == "YOLOv5RandomAffine"
    ds["dataset"]["pipeline"][4]["scaling_ratio_range"] = train_datasets_scaling_ratios[dataset_name]

    # Set CutOut number of holes and cutout shape (size) individually
    assert ds["dataset"]["pipeline"][5]["type"] == "mmdet.CutOut"
    ds["dataset"]["pipeline"][5]["n_holes"] = train_dataset_cutout_vals[dataset_name][0]
    ds["dataset"]["pipeline"][5]["cutout_shape"] = train_dataset_cutout_vals[dataset_name][1]

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
    dict(type='YOLOv5KeepRatioResize',
        scale=img_scale[1]), # width (or probably the bigger dimension) instead of img_scale to work properly. Figured out entirely by testing...
    dict(type='LetterResize',
        scale=img_scale[::-1], # This takes w*h, believe me... (or maybe it doesn't matter)
        allow_scale_up=False,
        pad_val=dict(img=pad_val)),
    dict(type='mmdet.PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor', 'pad_param'))
]
test_pipeline = val_pipeline

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
        pipeline = val_pipeline,
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
