import subprocess
from mmengine.config import Config

import paths


# Get default config
cfg = Config.fromfile(paths.model_config_filepath)

# Run subprocess
cmd = [paths.dist_train_script_filepath, paths.model_config_filepath, str(cfg.num_gpus)]
cmd += ["--work-dir", paths.working_dirpath]

opts = ["--cfg-options"]
# TODO allow --options from argv of this script?

if len(opts) == 1:
    opts = []

print(cmd)
ret = subprocess.call(cmd + opts)
