import os

proj_path = os.getcwd().replace("\\", "/") # Convert backslashes to forward slashes because windows sucks

# Parent folder of mmlabs repositories
mm_parent_dirpath = os.path.join(proj_path, "..")

working_dirpath = os.path.join(proj_path, "working_dir_yolox_mmyolo")

model_config_filename = "yolox_s_8xb8-300e_coco_custom.py"
model_config_filepath = os.path.join(mm_parent_dirpath, "mmyolo", "configs", "yolox", model_config_filename)

model_checkpoint_filename = "yolox_s_8xb8-300e_coco_20220917_030738-d7e60cb2.pth"
model_checkpoint_filepath = os.path.join(mm_parent_dirpath, "mmyolo", "checkpoints", model_checkpoint_filename)

last_checkpoint_link = os.path.join(working_dirpath, "last_checkpoint")
if os.path.exists(last_checkpoint_link):
    with open(last_checkpoint_link) as f:
        last_checkpoint_filepath = f.read()
else:
    last_checkpoint_filepath = None