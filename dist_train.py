import subprocess
from mmengine.config import Config

import paths


# TODO Take this as an argument?
num_gpus = 4

# Get default config
cfg = Config.fromfile(paths.model_config_filepath)

# Run subprocess
cmd = ["sh", paths.dist_train_script_filepath, paths.model_config_filepath, str(num_gpus)]
cmd += ["--work-dir", paths.working_dirpath]
if paths.last_checkpoint_filepath:
    cmd += ["--resume-from", paths.last_checkpoint_filepath]
cmd += ["--options", f"base_lr={cfg.base_lr * num_gpus}"]

# TODO allow --options from argv of this script?

ret = subprocess.call()