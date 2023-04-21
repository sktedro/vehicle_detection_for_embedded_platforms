"""
Converts model (with settings as in paths.py) to ONNX format to the working dir.
"""
import os
import sys
import argparse
from pprint import pprint
from mmdeploy.apis import torch2onnx

import sys
repo_path = os.path.join(os.path.dirname(__file__), '..')
sys.path.append(repo_path)
import paths


def main(args):
    assert os.path.exists(paths.deploy_config_filepath_onnx), f"Deploy config path ({paths.deploy_config_filepath_onnx}) not found"

    if isinstance(args.epoch, int):
        checkpoint_filepath = os.path.join(args.work_dir, "epoch_" + str(args.epoch) + ".pth")
    else:
        checkpoint_filepath = paths.get_last_checkpoint_filepath(args.work_dir)
    assert checkpoint_filepath and os.path.exists(checkpoint_filepath), f"Checkpoint path ({checkpoint_filepath}) not found"

    model_config_filepath = paths.get_config_from_working_dirpath(args.work_dir)

    print(args.work_dir)
    print(paths.deploy_config_filepath_onnx)
    print(model_config_filepath)
    print(checkpoint_filepath)

    print("Converting to ONNX")

    def torch2onnx_debug(**kwargs):
        pprint(kwargs)
        torch2onnx(**kwargs)

    torch2onnx_debug(
        img=os.path.join(repo_path, "deploy", "demo.jpg"),
        work_dir=args.work_dir,
        save_file=args.output_filename,
        deploy_cfg=paths.deploy_config_filepath_onnx,
        model_cfg=model_config_filepath,
        model_checkpoint=checkpoint_filepath,
        device="cpu"
        )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("work_dir",          type=str,   default=paths.working_dirpath, nargs="?",
                        help="working dirpath. Leave blank to use paths.working_dirpath")
    parser.add_argument("-o", "--output_filename",   type=str,   default=paths.deploy_onnx_filename,
                        help="output filename. Leave blank to use paths.deploy_onnx_filename")
    parser.add_argument("-e", "--epoch",     type=int,
                        help="epoch number to use. Leave blank to use latest")
    # TODO device arg?
    args = parser.parse_args()

    main(args)