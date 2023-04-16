"""
Converts model (with settings as in paths.py) to ONNX format to the working dir.
"""
import os
import os
import sys
import argparse
from pprint import pprint
from mmdeploy.apis import torch2onnx

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

def main(args):
    assert isinstance(args.epoch, int) or paths.last_checkpoint_filepath, "Epoch number not given and last checkpoint filepath empty"
    assert os.path.exists(paths.deploy_config_filepath), f"Deploy config path ({paths.deploy_config_filepath}) not found"
    assert os.path.exists(args.model_config), f"Model config path ({args.model_config}) not found"

    if isinstance(args.epoch, int):
        checkpoint_filepath = os.path.join(args.work_dir, "epoch_" + str(args.epoch) + ".pth")
    else:
        checkpoint_filepath = paths.last_checkpoint_filepath
    assert os.path.exists(checkpoint_filepath), f"Model checkpoint path ({checkpoint_filepath}) not found"

    print(args.work_dir)
    print(paths.deploy_config_filepath)
    print(args.model_config)
    print(checkpoint_filepath)

    print("Converting to ONNX")

    def torch2onnx_debug(**kwargs):
        pprint(kwargs)
        torch2onnx(**kwargs)

    torch2onnx_debug(
        img=os.path.join(repo_path, "deploy", "demo.jpg"),
        work_dir=args.work_dir,
        save_file=args.output_filename,
        deploy_cfg=paths.deploy_config_filepath,
        model_cfg=args.model_config,
        model_checkpoint=checkpoint_filepath,
        device="cpu"
        )


    # TODO convert onnx to ncnn
    # Doesn't work - get_onnx2ncnn_path() doesn't work...
    # print("Converting to NCNN")
    # onnx_path = os.path.join(args.work_dir, args.model_config)

    # save_param = args.model_config + '.param'
    # save_bin = args.model_config + '.bin'

    # onnx2ncnn_path = get_onnx2ncnn_path()
    # ret_code = call([onnx2ncnn_path, onnx_path, save_param, save_bin])
    # assert ret_code == 0, 'onnx2ncnn failed'

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("work_dir",          type=str,   default=paths.working_dirpath, nargs="?",
                        help="working dirpath. Leave blank to use paths.working_dirpath")
    parser.add_argument("model_config",      type=str,   default=paths.model_config_filepath, nargs="?",
                        help="model config filepath. Leave blank to use paths.model_config_filepath")
    parser.add_argument("-o", "--output_filename",   type=str,   default=paths.deploy_onnx_filename,
                        help="output filename. Leave blank to use paths.deploy_onnx_filename")
    parser.add_argument("-e", "--epoch",     type=int,
                        help="epoch number to use. Leave blank to use latest")
    args = parser.parse_args()

    main(args)