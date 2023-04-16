_base_ = [
    #  'mmyolo::_base_/default_runtime.py',
    #  'mmyolo::yolov8/yolov5_s-v61_syncbn_fast_8xb16-300e_coco.py',
    #  '/home/xskalo01/bp/proj/configs/yolov8_m.py',
]

# parameters that often need to be modified
num_classes = 80
img_scale = (640, 384)  # width, height
deepen_factor = 0.33
widen_factor = 0.5
max_epochs = 300
save_epoch_intervals = 10
train_batch_size_per_gpu = 16
train_num_workers = 8
val_batch_size_per_gpu = 1
val_num_workers = 2

# persistent_workers must be False if num_workers is 0.
persistent_workers = True

# Base learning rate for optim_wrapper
base_lr = 0.01

# only on Val
batch_shapes_cfg = dict(
    type='BatchShapePolicy',
    batch_size=val_batch_size_per_gpu,
    img_size=img_scale[0],
    size_divisor=32,
    extra_pad_ratio=0.5)

anchors = [
    [(10, 13), (16, 30), (33, 23)],  # P3/8
    [(30, 61), (62, 45), (59, 119)],  # P4/16
    [(116, 90), (156, 198), (373, 326)]  # P5/32
]
strides = [8, 16, 32]
num_det_layers = 3

# single-scale training is recommended to
# be turned on, which can speed up training.
env_cfg = dict(cudnn_benchmark=True)

architecture = dict(
    _scope_='mmyolo',
    type='YOLODetector',
    data_preprocessor=dict(  # 数据预处理器
        type='mmdet.DetDataPreprocessor',
        mean=[0., 0., 0.],
        std=[255., 255., 255.],
        bgr_to_rgb=True),
    backbone=dict(
        type='YOLOv5CSPDarknet',  # 主干网络
        deepen_factor=deepen_factor,
        widen_factor=widen_factor,
        norm_cfg=dict(type='BN', momentum=0.03, eps=0.001),
        act_cfg=dict(type='SiLU', inplace=True)),
    neck=dict(
        type='YOLOv5PAFPN',
        deepen_factor=deepen_factor,
        widen_factor=widen_factor,
        in_channels=[256, 512, 1024],
        out_channels=[256, 512, 1024],
        num_csp_blocks=3,
        norm_cfg=dict(type='BN', momentum=0.03, eps=0.001),
        act_cfg=dict(type='SiLU', inplace=True)),
    bbox_head=dict(
        type='YOLOv5Head',
        head_module=dict(
            type='YOLOv5HeadModule',
            num_classes=num_classes,
            in_channels=[256, 512, 1024],
            widen_factor=widen_factor,
            featmap_strides=strides,
            num_base_priors=3),
        prior_generator=dict(
            type='mmdet.YOLOAnchorGenerator',
            base_sizes=anchors,
            strides=strides),
        # scaled based on number of detection layers
        loss_cls=dict(
            type='mmdet.CrossEntropyLoss',
            use_sigmoid=True,
            reduction='mean',
            loss_weight=0.5 * (num_classes / 80 * 3 / num_det_layers)),
        loss_bbox=dict(
            type='IoULoss',
            iou_mode='ciou',
            bbox_format='xywh',
            eps=1e-7,
            reduction='mean',
            loss_weight=0.05 * (3 / num_det_layers),
            return_iou=True),
        loss_obj=dict(
            type='mmdet.CrossEntropyLoss',
            use_sigmoid=True,
            reduction='mean',
            loss_weight=1.0 * ((img_scale[0] / 640)**2 * 3 / num_det_layers)),
        prior_match_thr=4.,
        obj_level_weights=[4., 1., 0.4]),
    test_cfg=dict(
        multi_label=True,
        nms_pre=30000,
        score_thr=0.001,
        nms=dict(type='nms', iou_threshold=0.65),
        max_per_img=300))

default_scope='mmrazor'
model = dict(
    _scope_='mmrazor',
    type='DCFF',
    architecture=architecture,
    mutator_cfg=dict(
        type='DCFFChannelMutator',
        channel_unit_cfg=dict(
            type='DCFFChannelUnit', default_args=dict(choice_mode='ratio')),
        parse_cfg=dict(
            type='ChannelAnalyzer',
            demo_input=(1, 3, 224, 224),
            tracer_type='FxTracer')),
    # target_pruning_ratio=target_pruning_ratio,
    step_freq=1,
    linear_schedule=False)

model_wrapper = dict(
    type='mmcv.MMDistributedDataParallel', find_unused_parameters=True)

val_cfg = dict(_delete_=True, type='mmrazor.ItePruneValLoop')
