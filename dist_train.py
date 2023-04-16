import subprocess
import os
from mmengine.config import Config

import paths

# Note: to set port and IP (when training multiple models at the same time), set
# `export PORT=29501` and `export MASTER_ADDR=127.0.0.2` or something like that

def main():

    # Get default config
    cfg = Config.fromfile(paths.model_config_filepath)

    cmd = [os.path.join(paths.mmyolo_dirpath, "tools", "dist_train.sh")]
    cmd += [paths.model_config_filepath]
    cmd += [str(cfg.num_gpus)]
    cmd += ["--work-dir", paths.working_dirpath]

    # TODO allow --options from argv of this script?
    # opts = ["--cfg-options"]
    # if len(opts) > 1:
    #     cmd += opts

    print(cmd)
    subprocess.call(cmd)

    # This did not work, for some reason, though it should just be a nicer way to do
    # the same thing
    """
    from torch.distributed import run

    # Other params
    args = ["--nnodes", "1"]
    args += ["--node_rank", "1"]
    args += ["--master_addr", "127.0.0.1"]
    args += ["--master_port", "29500"]
    args += ["--nproc_per_node", str(cfg.num_gpus)]

    args += ["--no_python"]
    # args += ["--standalone"]

    # Config and train script
    args += [paths.model_config_filepath]
    args += [os.path.join(paths.mmyolo_dirpath, "tools", "train.py")]
    args += ["--work-dir", paths.working_dirpath]
    # args += ["--launcher", "pytorch"]
    # opts = ["--cfg-options"] # TODO allow options from argv of this script?
    # if len(opts) != 1:
    #     args += opts

    run.main(args)
    """

if __name__ == "__main__":
    main()
