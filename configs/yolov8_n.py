_base_ = './yolov8_m.py'

train_batch_size_per_gpu = 112 # YOLOv8-n, Sophie with 640x384. 120 -> CUDA out of memory

train_num_workers = 16 # On Sophie (internal GPU), more workers is better

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
