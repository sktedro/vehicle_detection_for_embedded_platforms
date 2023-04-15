_base_ = './yolov8_m.py'


train_batch_size_per_gpu = 224 # YOLOv8-n, Sophie with 512x288. Even at this point, CPU is 100% utilized with 4 GPUs

train_num_workers = 16

train_dataloader = dict(
    batch_size = train_batch_size_per_gpu,
    num_workers = train_num_workers)

optim_wrapper = dict(optimizer = dict(batch_size_per_gpu=train_batch_size_per_gpu))

# TODO these two: but what values?
deepen_factor = 0.166
widen_factor = 0.125
model = dict(
    backbone=dict(deepen_factor=deepen_factor, widen_factor=widen_factor),
    neck=dict(deepen_factor=deepen_factor, widen_factor=widen_factor),
    bbox_head=dict(head_module=dict(widen_factor=widen_factor)))
