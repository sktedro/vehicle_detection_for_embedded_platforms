_base_ = './yolov8_m.py'


train_batch_size_per_gpu = 256

train_num_workers = 16

train_dataloader = dict(
    batch_size = train_batch_size_per_gpu,
    num_workers = train_num_workers)

optim_wrapper = dict(optimizer = dict(batch_size_per_gpu=train_batch_size_per_gpu))

deepen_factor = 0.33
widen_factor = 0.25
model = dict(
    backbone=dict(deepen_factor=deepen_factor, widen_factor=widen_factor),
    neck=dict(deepen_factor=deepen_factor, widen_factor=widen_factor),
    bbox_head=dict(head_module=dict(widen_factor=widen_factor)))
