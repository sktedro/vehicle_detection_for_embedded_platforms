"""Settings for all configs to change individually per training session
"""

num_gpus = 2

#####
# img_scale - (width, height); need to be multiples of 32
# max_epochs - 300 is usually enough, medium or small could do with less, pico or smaller need 500
# base_lr - learning rate = 0.00125 generally, 0.01 for tiny models
#####


# YOLOv8-f 352x192
# img_scale = (352, 192)
# max_epochs = 500
# base_lr = 0.01

# YOLOv8-f 384x224
# img_scale = (384, 224)
# max_epochs = 500
# base_lr = 0.01

# YOLOv8-f 448x256
# img_scale = (448, 256)
# max_epochs = 500
# base_lr = 0.01

# YOLOv8-f 512x288
# img_scale = (512, 288)
# max_epochs = 500
# base_lr = 0.01

# YOLOv8 MobileNetV2 512x288
# img_scale = (512, 288)
# max_epochs = 300
# base_lr = 0.01

# YOLOv8-m 640x384 lr0.01
# img_scale = (640, 384)
# max_epochs = 300 # 300 # originally 500
# base_lr = 0.01

# YOLOv8-m 640x384
# img_scale = (640, 384)
# max_epochs = 300 # 300 # originally 500
# base_lr = 0.00125

# YOLOv8-n 448x256
# img_scale = (448, 256)
# max_epochs = 300
# base_lr = 0.00125

# YOLOv8-n 512x288
# img_scale = (512, 288)
# max_epochs = 300
# base_lr = 0.00125

# YOLOv8-n 640x384
# img_scale = (640, 384)
# max_epochs = 300
# base_lr = 0.00125

# YOLOv8-p 384x224
# img_scale = (384, 224)
# max_epochs = 500
# base_lr = 0.01

# YOLOv8-p 448x256
img_scale = (448, 256)
max_epochs = 500
base_lr = 0.01

# YOLOv8-p 512x288
# img_scale = (512, 288)
# max_epochs = 500
# base_lr = 0.01

# YOLOv8-s 640x384
# img_scale = (640, 384)
# max_epochs = 300
# base_lr = 0.00125


warmup_epochs = round(max_epochs // 100)
val_interval = 5 if max_epochs <= 300 else 10
save_epoch_intervals = val_interval