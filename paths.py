import os

proj_path = os.getcwd().replace("\\", "/") # Convert backslashes to forward slashes because windows sucks

# Parent folder of mmlabs repositories
mm_parent_dirpath = os.path.join(proj_path, "..")

working_dirpath = os.path.join(proj_path, "working_dir")

# model_config_filename = "yolov8_m_syncbn_fast_8xb16-500e_coco_custom.py"
model_config_filename = "yolov8_s_syncbn_fast_8xb16-500e_coco_custom.py"
model_config_filepath = os.path.join(mm_parent_dirpath, "mmyolo", "configs", "yolov8", model_config_filename)

model_checkpoint_filename = "yolov8_m_syncbn_fast_8xb16-500e_coco_20230115_192200-c22e560a.pth"
model_checkpoint_filepath = os.path.join(mm_parent_dirpath, "mmyolo", "checkpoints", model_checkpoint_filename)

last_checkpoint_link = os.path.join(working_dirpath, "last_checkpoint")
if os.path.exists(last_checkpoint_link):
    with open(last_checkpoint_link) as f:
        last_checkpoint_filepath = f.read()
else:
    last_checkpoint_filepath = None