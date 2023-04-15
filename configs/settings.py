"""Settings for all configs to change individually per training session
"""


num_gpus = 4


# img_scale = (384, 640) # height, width; need to be multiples of 32
img_scale = (288, 512) # For YOLOv8-n or YOLOv8-p 512x288


# YOLOv8-n 512x288
max_epochs = 300
warmup_epochs = 5
val_interval = 5

# YOLOv8-p 512x288
# max_epochs = 500
# warmup_epochs = 5
# val_interval = 10

# YOLOv8-m
# max_epochs = 300 # 300 # originally 500
# warmup_epochs = 3 # 3 # originally 3
# val_interval = 5 # default 10

save_epoch_intervals = val_interval