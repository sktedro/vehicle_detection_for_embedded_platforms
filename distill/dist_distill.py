import subprocess
import os
from mmengine.config import Config

import sys
repo_path = os.path.join(os.path.dirname(__file__), '..')
sys.path.append(repo_path)
import paths


train_script_filepath = os.path.join(paths.mmrazor_dirpath, "tools", "dist_train2.sh")

# Get default config
cfg = Config.fromfile(paths.distill_config_filepath)

cmd = [train_script_filepath]
cmd += [paths.model_config_filepath]
cmd += [str(cfg.num_gpus)]
cmd += ["--work-dir", paths.working_dirpath]

print(cmd)
subprocess.call(cmd)