"""Settings for all configs to change individually per training session
"""


num_gpus = 2


# img_scale - (height, width); need to be multiples of 32

# max_epochs - 300 is usually enough, medium or small could do with less, pico or smaller need 500

# Learning rate = 0.00125 per gpu, linear to batch size (https://stackoverflow.com/questions/53033556/how-should-the-learning-rate-change-as-the-batch-size-change)
# Per gpu, because mmengine anyways says when training: LR is set based on batch size of [batch_size*num_gpus] and the current batch size is [batch_size]. Scaling the original LR by [1/num_gpus].
# base_lr = 0.00125 * num_gpus * (train_batch_size_per_gpu / pre_trained_model_batch_size_per_gpu)
# base_lr = 0.00125 # Tried different LRs on 4 GPUs, 48 batch size, and 0.002 was worse and 0.0005 was worse, so scaling doesn't seem to be needed...
# base_lr = 0.01 # Default


# YOLOv8-m 640x384
# img_scale = (384, 640)
# max_epochs = 300 # 300 # originally 500
# base_lr = 0.00125

# YOLOv8-s 640x384
# img_scale = (384, 640)
# max_epochs = 300
# base_lr = 0.00125

# YOLOv8-n 640x384
# img_scale = (384, 640)
# max_epochs = 300
# base_lr = 0.00125

# YOLOv8-n 512x288
# img_scale = (288, 512)
# max_epochs = 300
# base_lr = 0.00125

# TODO YOLOv8-n 448x256
# img_scale = (256, 448)
# max_epochs = 300
# base_lr = 0.00125

# YOLOv8-p 512x288
# img_scale = (288, 512)
# max_epochs = 500
# base_lr = 0.01

# TODO YOLOv8-p 448x256
img_scale = (256, 448)
max_epochs = 500
base_lr = 0.01

# YOLOv8-p 384x224
# img_scale = (224, 384)
# max_epochs = 500
# base_lr = 0.01

# YOLOv8-f 512x288
# img_scale = (288, 512)
# max_epochs = 500
# base_lr = 0.01

# YOLOv8-f 448x256
# img_scale = (256, 448)
# max_epochs = 500
# base_lr = 0.01

# YOLOv8-f 384x224
# img_scale = (224, 384)
# max_epochs = 500
# base_lr = 0.01

# YOLOv8-f 352x192
# img_scale = (192, 352)
# max_epochs = 500
# base_lr = 0.01


# YOLOv8 MobileNetV2 512x288
# img_scale = (288, 512)
# max_epochs = 300
# base_lr = 0.01


warmup_epochs = round(max_epochs // 100)
val_interval = 5 if max_epochs <= 300 else 10
save_epoch_intervals = val_interval