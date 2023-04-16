"""Settings for all configs to change individually per training session
"""


num_gpus = 2


# img_scale = (384, 640) # height, width; need to be multiples of 32
img_scale = (288, 512) # For YOLOv8-n or YOLOv8-p 512x288


# Learning rate = 0.00125 per gpu, linear to batch size (https://stackoverflow.com/questions/53033556/how-should-the-learning-rate-change-as-the-batch-size-change)
# Per gpu, because mmengine anyways says when training: LR is set based on batch size of [batch_size*num_gpus] and the current batch size is [batch_size]. Scaling the original LR by [1/num_gpus].
# base_lr = 0.00125 * num_gpus * (train_batch_size_per_gpu / pre_trained_model_batch_size_per_gpu)
# base_lr = 0.00125 # Tried different LRs on 4 GPUs, 48 batch size, and 0.002 was worse and 0.0005 was worse, so scaling doesn't seem to be needed...
# base_lr = 0.01 # Default

# YOLOv8 MobileNetV2 512x288
max_epochs = 300
warmup_epochs = 5
val_interval = 5
base_lr = 0.01

# YOLOv8-n 512x288
# max_epochs = 300
# warmup_epochs = 5
# val_interval = 5

# YOLOv8-p 512x288
# max_epochs = 500
# warmup_epochs = 5
# val_interval = 10

# YOLOv8-m
# max_epochs = 300 # 300 # originally 500
# warmup_epochs = 3 # 3 # originally 3
# val_interval = 5 # default 10
# base_lr = 0.00125

save_epoch_intervals = val_interval