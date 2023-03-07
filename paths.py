import os

##### BASE PATHS #####

try:
    proj_path = os.path.abspath(__file__)
except:
    proj_path = os.path.abspath("__file__")
proj_path = os.path.dirname(proj_path).replace("\\", "/") # Convert backslashes to forward slashes because windows sucks

datasets_dirpath = "/home/tedro/Downloads/datasets/"
# datasets_dirpath = "/Users/z004ktej/Downloads/datasets/"
# datasets_dirpath = "/home/xskalo01/datasets/"

mmyolo_dirpath = os.path.join(proj_path, "..", "mmyolo")
mmdeploy_dirpath = os.path.join(proj_path, "..", "mmdeploy")
mmrazor_dirpath = os.path.join(proj_path, "..", "mmrazor")

working_dirpath = os.path.join(proj_path, "working_dir_yolov8_dist_384_conf5")


##### CONFIG #####

model_config_filename = "yolov8_m.py"
# model_config_filename = "yolov8_n.py"
model_config_filepath = os.path.join(proj_path, "configs", model_config_filename)


##### CHECKPOINT #####

model_checkpoint_filename = "yolov8_m_syncbn_fast_8xb16-500e_coco_20230115_192200-c22e560a.pth"
# model_checkpoint_filename = "yolov8_n_syncbn_fast_8xb16-500e_coco_20230114_131804-88c11cdb.pth"
model_checkpoint_filepath = os.path.join(proj_path, "checkpoints", model_checkpoint_filename)


##### DISTILL #####

distill_config_filepath = os.path.join(proj_path, "distill", "conf.py")

##### DEPLOY #####

deploy_config_filename = "detection_onnxruntime_static.py"
deploy_config_filepath = os.path.join(proj_path, "deploy", deploy_config_filename)
deploy_onnx_filename = "end2end.onnx"


##### AUTOMATION #####

last_checkpoint_link = os.path.join(working_dirpath, "last_checkpoint")
if os.path.exists(last_checkpoint_link):
    with open(last_checkpoint_link) as f:
        last_checkpoint_filepath = f.read().replace("\n", "").replace("\r", "")
else:
    last_checkpoint_filepath = None

# Check that everything is in the right place

assert os.path.exists(proj_path), f"Project path ({proj_path}) not found"
assert os.path.exists(datasets_dirpath), f"Datasets path ({datasets_dirpath}) not found"

assert os.path.exists(model_config_filepath), f"Model config path ({model_config_filepath}) not found"
assert os.path.exists(model_checkpoint_filepath), f"Model checkpoint path ({model_checkpoint_filepath}) not found"
