"""Prints model structure using a config file"""
import os
import argparse
import sys
from contextlib import contextmanager
from mmengine.runner import Runner
from mmengine.config import Config

import paths


@contextmanager
def suppress_stdout():
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout


def main(args):
    if os.path.isabs(args.config_filepath):
        pass
    elif os.path.exists(os.path.join(paths.proj_path, args.config_filepath)):
        args.config_filepath = os.path.join(paths.proj_path, args.config_filepath)
    elif os.path.exists(os.path.join(paths.proj_path, "configs", args.config_filepath)):
        args.config_filepath = os.path.join(paths.proj_path, "configs", args.config_filepath)
    else:
        print("FATAL: Config file was not found in any of these locations:")
        if os.path.isabs(args.config_filepath):
            print(args.config_filepath)
        print(os.path.join(paths.proj_path, args.config_filepath))
        print(os.path.join(paths.proj_path, "configs", args.config_filepath))
        return

    print(f"Model structure of {os.path.basename(args.config_filepath)}:")

    with suppress_stdout():
        cfg = Config.fromfile(args.config_filepath)
        model = Runner.from_cfg(cfg).model
    print(model.__str__())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config_filepath", type=str, default=paths.model_config_filepath, nargs="?",
                        help="config filepath. Leave blank to use one from paths.py. If the path is not absolute, resolving is attempted both relative to the project path and relative to the configs folder")
    args = parser.parse_args()

    main(args)
