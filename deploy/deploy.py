"""
Converts model (with settings as in paths.py) to ONNX format to the working dir.
"""
import os
from mmdeploy.apis import torch2onnx

import sys
repo_path = os.path.join(os.path.dirname(__file__), '..')
sys.path.append(repo_path)
import paths


def main():
    assert paths.last_checkpoint_filepath != None
    assert os.path.exists(paths.deploy_config_filepath), f"Deploy config path ({paths.deploy_config_filepath}) not found"
    assert os.path.exists(paths.model_config_filepath), f"Model config path ({paths.model_config_filepath}) not found"
    assert os.path.exists(paths.last_checkpoint_filepath), f"Model checkpoint path ({paths.last_checkpoint_filepath}) not found"

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