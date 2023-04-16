_base_ = './yolov8_m.py'

# Taken from mmdet/models/backbones/mobilenet_v2.py:
# Parameters to build layers. 4 parameters are needed to construct a
# layer, from left to right: expand_ratio, channel, num_blocks, stride.
# arch_settings = [[1, 16, 1, 1], [6, 24, 2, 2], [6, 32, 3, 2],
#                      [6, 64, 4, 2], [6, 96, 3, 1], [6, 160, 3, 2],
#                      [6, 320, 1, 1]]

mobilenet_out_indices = (2, 4, 6)
deepen_factor = 1
widen_factor = 1
channels = [32, 96, 320]
train_batch_size_per_gpu = 96 # 128 too much

# TODO
# mobilenet_out_indices = (4, 7)
# deepen_factor = 1
# widen_factor = 1
# channels = [32, 96, 320]
# train_batch_size_per_gpu = 96 # 128 too much

train_num_workers = 16

train_dataloader = dict(
    batch_size = train_batch_size_per_gpu,
    num_workers = train_num_workers)

optim_wrapper = dict(optimizer = dict(batch_size_per_gpu=train_batch_size_per_gpu))

model = dict(
    backbone=dict(
        _delete_=True,
        type='mmdet.MobileNetV2',
        out_indices=mobilenet_out_indices,
        act_cfg=dict(type='LeakyReLU', negative_slope=0.1),
        init_cfg=dict(
            type='Pretrained',
            checkpoint='/home/xskalo01/bp/proj/checkpoints/mobilenet_v2_batch256_imagenet_20200708-3b2dc3af.pth'
            )
        ),
    neck=dict(
        in_channels=channels,
        out_channels=channels,
        deepen_factor=deepen_factor,
        widen_factor=widen_factor
        ),
    bbox_head=dict(
        head_module=dict(
            in_channels=channels,
            widen_factor=widen_factor
            )
        )
    )

find_unused_parameters = True # To allow for distributed training
# https://github.com/pytorch/pytorch/issues/43259
# https://github.com/open-mmlab/mmdetection/issues/7298