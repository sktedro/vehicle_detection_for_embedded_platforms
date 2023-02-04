import os

proj_path = os.getcwd().replace("\\", "/") # Convert backslashes to forward slashes because windows sucks

drive_dirpath = os.path.join(proj_path, "drive")
drive_traffic_cams_path = os.path.join(drive_dirpath, "traffic_cams")

mmdetection_path = os.path.join(proj_path, "..", "mmdetection")

working_dirpath = os.path.join(proj_path, "working_dir_long_all")

process_dataset_dirpath = os.path.join(proj_path, "process_dataset")


mmdetection_config_path = os.path.join(mmdetection_path, "configs")
mmdetection_checkpoint_path = os.path.join(mmdetection_path, "checkpoints")

# model_config_filename = "yolox_x_8x8_300e_coco.py"
# model_config_filename = "yolox_s_8x8_300e_coco.py"
model_config_filename = "yolox_s_8x8_300e_coco_custom.py"
# model_config_filename = "yolox_tiny_8x8_300e_coco.py"
# model_config_filename = "yolox_nano_8x8_300e_coco.py"
model_config_filepath = os.path.join(mmdetection_config_path, "yolox", model_config_filename)

# And YOLOX-x checkpoint
# model_checkpoint_filename = "yolox_x_8x8_300e_coco_20211126_140254-1ef88d67.pth"
model_checkpoint_filename = "yolox_s_8x8_300e_coco_20211121_095711-4592a793.pth"
# model_checkpoint_filename = "yolox_tiny_8x8_300e_coco_20211124_171234-b4047906.pth"
# model_checkpoint_filename = "yolox_nano.pth"
model_checkpoint_filepath = os.path.join(mmdetection_path, "checkpoints", model_checkpoint_filename)
if not os.path.exists(mmdetection_checkpoint_path):
    os.mkdir(mmdetection_checkpoint_path)


last_checkpoint_filepath = os.path.join(working_dirpath, "latest.pth")
if not os.path.exists(last_checkpoint_filepath):
    last_checkpoint_filepath = None

saved_model_filepath = os.path.join(working_dirpath, 'model.pickle')
saved_config_filepath = os.path.join(working_dirpath, 'config.pickle')