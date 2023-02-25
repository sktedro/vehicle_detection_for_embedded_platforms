from mmdeploy.apis import torch2onnx

import sys, os
repo_path = os.path.join(os.path.dirname(__file__), '..')
sys.path.append(repo_path)
import paths

assert paths.last_checkpoint_filepath != None
assert os.path.exists(paths.deploy_config_filepath)
assert os.path.exists(paths.model_config_filepath)
assert os.path.exists(paths.last_checkpoint_filepath)

img = os.path.join(repo_path, "deploy", "demo.jpg")
work_dir = paths.working_dirpath
save_file = paths.deploy_onnx_filename
deploy_cfg = paths.deploy_config_filepath
model_cfg = paths.model_config_filepath
model_checkpoint = paths.last_checkpoint_filepath
device = "cpu"

torch2onnx(img, work_dir, save_file, deploy_cfg, model_cfg, model_checkpoint, device)