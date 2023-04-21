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
working_dirname = "working_dir_yolov8_f_conf8_512x288_lr01"
model_config_filename = "yolov8_f_512x288.py"
model_checkpoint_filename = None

# TODO YOLOv8-f 448x256
# working_dirname = "working_dir_yolov8_f_conf8_448x256_lr01"
# model_config_filename = "yolov8_f_448x256.py"
# model_checkpoint_filename = None

# TODO YOLOv8-f 384x224
# working_dirname = "working_dir_yolov8_f_conf8_384x224_lr01"
# model_config_filename = "yolov8_f_384x224.py"
# model_checkpoint_filename = None




working_dirpath = os.path.join(proj_path, working_dirname)

model_config_filepath = os.path.join(proj_path, "configs", model_config_filename)

if model_checkpoint_filename:
    model_checkpoint_filepath = os.path.join(proj_path, "checkpoints", model_checkpoint_filename)
else:
    model_checkpoint_filepath = None


##### DISTILL #####

distill_config_filepath = os.path.join(proj_path, "distill", "conf.py")

##### DEPLOY #####

deploy_config_filepath_onnx = os.path.join(proj_path, "deploy", "detection_onnxruntime_static.py")
deploy_config_filepath_trt = os.path.join(proj_path, "deploy", "detection_tensorrt_static-640x640.py")
deploy_config_filepath_ts = os.path.join(proj_path, "deploy", "detection_torchscript.py")
deploy_onnx_filename = "end2end.onnx"
deploy_trt_filename = "end2end.trt"
deploy_armnn_filename = "end2end.armnn"
deploy_openvino_filename = "end2end.openvino"
deploy_ncnn_filename = "end2end"


##### AUTOMATION #####

def get_last_checkpoint_filepath(working_dirpath):
    """ Takes a working dir path as an argument and returns the filepath of last
    checkpoint in the working dir, based on the `last_checkpoint` file in the
    directory. If not found, raises an Exception """
    last_checkpoint_link = os.path.join(working_dirpath, "last_checkpoint")
    if os.path.exists(last_checkpoint_link):
        with open(last_checkpoint_link) as f:
            last_checkpoint_filepath = f.read().replace("\n", "").replace("\r", "")
            # The absolute path depends on the machine, so make it work anywhere:
            last_checkpoint_filename = os.path.basename(last_checkpoint_filepath)
            return os.path.join(working_dirpath, last_checkpoint_filename)
    else:
        raise Exception(f"FATAL: No last checkpoint found in the working dir {working_dirpath}")

def get_best_checkpoint_filepath(working_dirpath):
    """ Takes a working dir path as an argument and returns a filepath to the
    best checkpoint based on COCO metric. If there are multiple `best_coco`
    files, it returns the latest (in case of multiple training runs). If there
    is no best checkpoint, raises and Exception """
    best_coco_dirpath = os.path.join(working_dirpath, "best_coco")
    if os.path.isdir(best_coco_dirpath):
        files = os.listdir(best_coco_dirpath)
        files_by_epoch = {}
        for f in files:
            epoch_number = int(f.split("_")[-1].split(".")[0])
            files_by_epoch[epoch_number] = f
        try:
            best_epoch_number = max(list(files_by_epoch.keys()))
        except ValueError:
            raise Exception(f"FATAL: No best checkpoint found in the working dir {working_dirpath}")
        return os.path.join(working_dirpath,
                            "best_coco",
                            files_by_epoch[best_epoch_number])
    raise Exception(f"FATAL: No best checkpoint found in the working dir {working_dirpath}")


def get_config_from_working_dirpath(working_dirpath):
    """ Takes a working dir path as an argument and returns a filepath to the
    training config file. If there are multiple python files in the working dir,
    it filters out filenames that are not in configs/. If there is still more
    than one left, Exception is raised """
    files = [f for f in os.listdir(working_dirpath) if f.endswith("py")]
    if len(files) == 1:
        return os.path.join(working_dirpath, files[0])
    elif len(files) == 0:
        raise Exception(f"FATAL: No config file was found in the working dir {working_dirpath}")
    else:
        # If only one is name can be found in configs, choose it
        available_config_files = os.listdir(os.path.join(proj_path, "configs"))
        files_filtered = [f for f in files if f in available_config_files]
        if len(files_filtered) == 1:
            print(f"Choosing {files_filtered[0]} from {files} as the config file")
            return os.path.join(working_dirpath, files[0])
        else:
            raise Exception(f"FATAL: More than one python file found in the working dir: {files}")


last_checkpoint_filepath = get_last_checkpoint_filepath(working_dirpath)

# Check that everything is in the right place

for path in [proj_path, datasets_dirpath, model_config_filepath, model_checkpoint_filepath]:
    if not path or not os.path.exists(path):
        print(f"WARNING: paths.py: Path {path} does not exist")