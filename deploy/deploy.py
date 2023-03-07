"""
Converts model (with settings as in paths.py) to ONNX format to the working dir.
"""
import os
from mmdeploy.apis import torch2onnx
import os
import sys

import sys
repo_path = os.path.join(os.path.dirname(__file__), '..')
sys.path.append(repo_path)
import paths



# Or we can use the mmyolo's easydeploy script export.py
# TODO how to execute it? Rely on "python3"?
# def main():
#     export_script_filepath = os.path.join(paths.mmyolo_dirpath, "projects", "easydeploy", "tools", "export.py")

#     cmd = []
#     cmd += [export_script_filepath]
#     cmd += [paths.model_config_filepath]
#     cmd += [paths.last_checkpoint_filepath]
#     cmd += ["--work-dir", paths.working_dirpath]
#     cmd += ["--img-size", 640, 384] # TODO read from config
#     cmd += ["--batch", 1]
#     cmd += ["--device", "cpu"]
#     cmd += ["--simplify"]
#     cmd += ["--opset", 11]
#     cmd += ["--backend", 1]
#     cmd += ["--pre-topk", 1000]
#     cmd += ["--keep-topk", 100]
#     cmd += ["--iou-threshold", 0.65]
#     cmd += ["--score-threshold", 0.25]

#     for i in range(len(cmd)):
#         if not isinstance(cmd[i], str):
#             cmd[i] = str(cmd[i])
#     print(" ".join(cmd))

def main():
    assert paths.last_checkpoint_filepath != None
    assert os.path.exists(paths.deploy_config_filepath), f"Deploy config path ({paths.deploy_config_filepath}) not found"
    assert os.path.exists(paths.model_config_filepath), f"Model config path ({paths.model_config_filepath}) not found"
    assert os.path.exists(paths.last_checkpoint_filepath), f"Model checkpoint path ({paths.last_checkpoint_filepath}) not found"

    print(paths.working_dirpath)
    print(paths.deploy_config_filepath)
    print(paths.model_config_filepath)
    print(paths.last_checkpoint_filepath)

    torch2onnx(
        img=os.path.join(repo_path, "deploy", "demo.jpg"),
        work_dir=paths.working_dirpath,
        save_file=paths.deploy_onnx_filename,
        deploy_cfg=paths.deploy_config_filepath,
        model_cfg=paths.model_config_filepath,
        model_checkpoint=paths.last_checkpoint_filepath,
        device="cpu"
        )

if __name__ == "__main__":
    main()