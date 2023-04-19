import os

# TODO model_checkpoint_filename should be in the model config
# TODO stop using settings.py and somehow put the contents into configs or something

##### BASE PATHS #####

try:
    proj_path = os.path.abspath(__file__)
except:
    proj_path = os.path.abspath("__file__")
proj_path = os.path.dirname(proj_path).replace("\\", "/") # Convert backslashes to forward slashes because windows sucks

# datasets_dirpath = "/home/tedro/Downloads/datasets/"
# datasets_dirpath = "/Users/z004ktej/Downloads/datasets/"
datasets_dirpath = "/home/xskalo01/datasets/"

mmyolo_dirpath = os.path.join(proj_path, "..", "mmyolo")
mmdeploy_dirpath = os.path.join(proj_path, "..", "mmdeploy")
mmrazor_dirpath = os.path.join(proj_path, "..", "mmrazor")


##### TODO WORK DIR #####

# Don't forget to update settings.py when changing the model config

# TODO YOLOv8 with MobileNetV2 (indices 2,4,6)
# working_dirname = "working_dir_yolov8_conf8_512x288_mobilenet_v2_indices_246"
# model_config_filename = "yolov8_512x288_mobilenet_v2_indices_246.py"
# model_checkpoint_filename = None

# TODO YOLOv8-s
# working_dirname = "working_dir_yolov8_s_conf8"
# model_config_filename = "yolov8_s.py"
# model_checkpoint_filename = "yolov8_s_syncbn_fast_8xb16-500e_coco_20230117_180101-5aa5f0f1.pth"

# TODO YOLOv8-n 448x256
# working_dirname = "working_dir_yolov8_n_conf8_448x256"
# model_config_filename = "yolov8_n_448x256.py"
# model_checkpoint_filename = "yolov8_n_syncbn_fast_8xb16-500e_coco_20230114_131804-88c11cdb.pth"

# TODO YOLOv8-p 448x256
# working_dirname = "working_dir_yolov8_p_conf8_448x256_lr01"
# model_config_filename = "yolov8_p_448x256.py"
# model_checkpoint_filename = None

# TODO YOLOv8-p 384x224
# working_dirname = "working_dir_yolov8_p_conf8_384x224_lr01"
# model_config_filename = "yolov8_p_384x224.py"
# model_checkpoint_filename = None

# TODO YOLOv8-f 512x288
# working_dirname = "working_dir_yolov8_f_conf8_512x288_lr01"
# model_config_filename = "yolov8_f_512x288.py"
# model_checkpoint_filename = None

# TODO YOLOv8-f 448x256
# working_dirname = "working_dir_yolov8_f_conf8_448x256_lr01"
# model_config_filename = "yolov8_f_448x256.py"
# model_checkpoint_filename = None

# TODO YOLOv8-f 384x224
working_dirname = "working_dir_yolov8_f_conf8_384x224_lr01"
model_config_filename = "yolov8_f_384x224.py"
model_checkpoint_filename = None




working_dirpath = os.path.join(proj_path, working_dirname)

model_config_filepath = os.path.join(proj_path, "configs", model_config_filename)

if model_checkpoint_filename:
    model_checkpoint_filepath = os.path.join(proj_path, "checkpoints", model_checkpoint_filename)
else:
    model_checkpoint_filepath = None


##### DISTILL #####

distill_config_filepath = os.path.join(proj_path, "distill", "conf.py")

##### DEPLOY #####

deploy_config_filename = "detection_onnxruntime_static.py"
deploy_config_filepath = os.path.join(proj_path, "deploy", deploy_config_filename)
deploy_onnx_filename = "end2end.onnx"
deploy_trt_filename = "end2end.trt"
deploy_armnn_filename = "end2end.armnn"
deploy_openvino_filename = "end2end.openvino"
deploy_ncnn_filename = "end2end"


##### AUTOMATION #####

last_checkpoint_link = os.path.join(working_dirpath, "last_checkpoint")
if os.path.exists(last_checkpoint_link):
    with open(last_checkpoint_link) as f:
        last_checkpoint_filepath = f.read().replace("\n", "").replace("\r", "")
        # The absolute path depends on the machine, so make it work anywhere:
        last_checkpoint_filename = os.path.basename(last_checkpoint_filepath)
        last_checkpoint_filepath = os.path.join(working_dirpath, last_checkpoint_filename)
else:
    last_checkpoint_filepath = None


# Check that everything is in the right place

for path in [proj_path, datasets_dirpath, model_config_filepath, model_checkpoint_filepath]:
    if not path or not os.path.exists(path):
        print(f"WARNING: paths.py: Path {path} does not exist")