import os


proj_path = os.getcwd().replace("\\", "/") # Convert backslashes to forward slashes because windows sucks

# Paths: (absolute path recommended)
# datasets_dirpath = "/home/tedro/Downloads/datasets/"
# datasets_dirpath = "/Users/z004ktej/Downloads/datasets/"
datasets_dirpath = "/home/xskalo01/datasets/"


dist_train_script_filepath = os.path.join("..", "mmyolo", "tools", "dist_train.sh")


working_dirpath = os.path.join(proj_path, "working_dir_yolov8_dist")
# working_dirpath = os.path.join(proj_path, "working_dir_yolov8_n")

model_config_filename = "yolov8_m_syncbn_fast_8xb16-500e_coco_custom.py"
# model_config_filename = "yolov8_n_syncbn_fast_8xb16-500e_coco_custom.py"
model_config_filepath = os.path.join(proj_path, "configs", model_config_filename)

model_checkpoint_filename = "yolov8_m_syncbn_fast_8xb16-500e_coco_20230115_192200-c22e560a.pth"
# model_checkpoint_filename = "yolov8_n_syncbn_fast_8xb16-500e_coco_20230114_131804-88c11cdb.pth"
model_checkpoint_filepath = os.path.join(proj_path, "checkpoints", model_checkpoint_filename)


last_checkpoint_link = os.path.join(working_dirpath, "last_checkpoint")
if os.path.exists(last_checkpoint_link):
    with open(last_checkpoint_link) as f:
        last_checkpoint_filepath = f.read()
else:
    last_checkpoint_filepath = None


# Assert everything is in the right place
assert os.path.exists(proj_path)
assert os.path.exists(datasets_dirpath)

assert os.path.exists(model_config_filepath)
assert os.path.exists(model_checkpoint_filepath)
