_base_ = './yolov8_m.py'

train_batch_size_per_gpu = 128 # 128 TODO

train_num_workers = 16

train_dataloader = dict(
    batch_size = train_batch_size_per_gpu,
    num_workers = train_num_workers)

optim_wrapper = dict(optimizer = dict(batch_size_per_gpu=train_batch_size_per_gpu))

# Taken from mmdet/models/backbones/mobilenet_v2.py:
# Parameters to build layers. 4 parameters are needed to construct a
# layer, from left to right: expand_ratio, channel, num_blocks, stride.
# arch_settings = [[1, 16, 1, 1], [6, 24, 2, 2], [6, 32, 3, 2],
#                      [6, 64, 4, 2], [6, 96, 3, 1], [6, 160, 3, 2],
#                      [6, 320, 1, 1]]

# Changing input resolution does nothing to channels

deepen_factor = 0.33
widen_factor = 0.25
# channels=[320, 96, 32]
# channels = [32, 96, 320]
model = dict(
    backbone=dict(
        _delete_=True,
        type='mmdet.MobileNetV2',
        out_indices=(2, 4, 6), # Changing order does nothing
        # out_indices=(4, 7),
        act_cfg=dict(type='LeakyReLU', negative_slope=0.1),
        # init_cfg=dict(
        #     type='Pretrained',
            # checkpoint='open-mmlab://mmdet/mobilenet_v2'
            # checkpoint='../checkpoints/mobilenet_v2_batch256_imagenet_20200708-3b2dc3af.pth'
            # checkpoint='/home/xskalo01/bp/proj/checkpoints/mobilenet_v2_batch256_imagenet_20200708-3b2dc3af.pth'
        #     )
        ),
    neck=dict(
        in_channels=[320, 96, 32], # expected 32, got 416
        # in_channels=[32, 96, 320], # expected 104, got 416

        # in_channels=[96, 1280],

        # Setting the channels everywhere does nothing
        # in_channels = channels,
        # out_channels = channels,

        # in_channels=[96, 320, 1280], # expected 400, got 416
        # in_channels=[96, 320, 1344], # passes and then expected 104, got 160
        # in_channels=[256, 320, 1344], # passes and then expected 144, got 160
        # in_channels=[320, 320, 1344], # passes 2x and then expected 80, got 64

        # in_channels=[96, 320, 1280], # expected 400, got 416
        # in_channels=[96, 512, 1280], # expected 448, got 416
        # in_channels=[96, 360, 1280], # passes and then expected 120, got 160
        # in_channels=[64, 360, 1280], # passes and then expected 112, got 160
        # in_channels=[256, 360, 1280], # passes 2x and then expected 96, got 128, pravdepodobne v CSPLayerWithTwoConv (1)

        # in_channels=[96, 320, 1536], # expected 464, got 416
        # in_channels=[96, 320, 2560],
        # in_channels=[24, 32, 96], # Povedal GPT, nefunguje
        deepen_factor=deepen_factor,
        widen_factor=widen_factor
        ),
    bbox_head=dict(
        head_module=dict(
            widen_factor=widen_factor,
            # in_channels = channels
            )
        )
    )