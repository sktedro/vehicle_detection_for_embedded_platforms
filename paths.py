import os

proj_path = os.getcwd().replace("\\", "/") # Convert backslashes to forward slashes because windows sucks

drive_dirpath = os.path.join(proj_path, "drive")
drive_traffic_cams_path = os.path.join(drive_dirpath, "traffic_cams")

mmdetection_path = os.path.join(proj_path, "..", "mmdetection")

working_dirpath = os.path.join(proj_path, "working_dir")

process_dataset_dirpath = os.path.join(proj_path, "process_dataset")


mmdetection_config_path = os.path.join(mmdetection_path, "configs")
mmdetection_checkpoint_path = os.path.join(mmdetection_path, "checkpoints")

# model_config_filename = "yolox_s_8x8_300e_coco.py"
model_config_filename = "yolox_s_8xb8-300e_coco_custom.py"
# model_config_filename = "yolox_tiny_8x8_300e_coco.py"
# model_config_filename = "yolox_nano_8x8_300e_coco.py"
model_config_filepath = os.path.join(mmdetection_config_path, "yolox", model_config_filename)
# model_config_filepath = os.path.join(mmdetection_config_path, "yolov8", "yolov8_m_syncbn_fast_8xb16-500e_coco_custom.py")

# And YOLOX-x checkpoint
model_checkpoint_filename = "yolox_s_8x8_300e_coco_20211121_095711-4592a793.pth"
# model_checkpoint_filename = "yolox_tiny_8x8_300e_coco_20211124_171234-b4047906.pth"
# model_checkpoint_filename = "yolov8_m_syncbn_fast_8xb16-500e_coco_20230115_192200-c22e560a.pth"
# model_checkpoint_filename = "yolox_nano.pth"
model_checkpoint_filepath = os.path.join(mmdetection_path, "checkpoints", model_checkpoint_filename)
if not os.path.exists(mmdetection_checkpoint_path):
    os.mkdir(mmdetection_checkpoint_path)

# mmdet v2.0
# last_checkpoint_filepath = os.path.join(working_dirpath, "latest.pth")
# if not os.path.exists(last_checkpoint_filepath):
#     last_checkpoint_filepath = None

last_checkpoint_link = os.path.join(working_dirpath, "last_checkpoint")
if os.path.exists(last_checkpoint_link):
    with open(last_checkpoint_link) as f:
        last_checkpoint_filepath = f.read()
else:
    last_checkpoint_filepath = None
    