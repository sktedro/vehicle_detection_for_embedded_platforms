import argparse
import os
from pprint import pprint

from mmdet.apis import init_detector
from mmdeploy.apis import torch2torchscript # TODO use this!
from mmyolo.utils import register_all_modules
register_all_modules()

import sys
repo_path = os.path.join(os.path.dirname(__file__), '..')
sys.path.append(repo_path)
import paths


def main(args):

    if isinstance(args.epoch, int):
        checkpoint_filepath = os.path.join(args.work_dir, "epoch_" + str(args.epoch) + ".pth")
    else:
        checkpoint_filepath = paths.get_last_checkpoint_filepath(args.work_dir)
    assert checkpoint_filepath and os.path.exists(checkpoint_filepath), f"Checkpoint path ({checkpoint_filepath}) not found"

    model_config_filepath = paths.get_config_from_working_dirpath(args.work_dir)

    print(args.work_dir)
    print(paths.deploy_config_filepath_ts)
    print(model_config_filepath)
    print(checkpoint_filepath)

    model = init_detector(model_config_filepath, checkpoint_filepath, device="cpu")

    if checkpoint_filepath.endswith(".pth"):
        output_filepath = checkpoint_filepath[:-1] # Same name without `h` in `.pth`
    else:
        raise Exception(f"Checkpoint filename should have the .pth extension ({checkpoint_filepath})")

    print("Converting to TorchScript")

    # TODO Save as TorchScript
    def torch2ts_debug(**kwargs):
        pprint(kwargs)
        torch2torchscript(**kwargs)

    torch2ts_debug(
        img=os.path.join(repo_path, "deploy", "demo.jpg"),
        work_dir=args.work_dir,
        save_file=output_filepath,
        deploy_cfg=paths.deploy_config_filepath_ts,
        model_cfg=model_config_filepath,
        model_checkpoint=checkpoint_filepath,
        device="cpu"
        )

    # This won't work because the model is not a PyTorch model
    # model_scripted = torch.jit.script(model)
    # model_scripted.save(output_filepath)

    print(f"Saved to {output_filepath}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("work_dir",          type=str,   default=paths.working_dirpath, nargs="?",
                        help="working dirpath. Leave blank to use paths.working_dirpath")
    parser.add_argument("-e", "--epoch",     type=int,
                        help="epoch number to use. Leave blank to use latest")
    # TODO device arg?
    args = parser.parse_args()

    main(args)