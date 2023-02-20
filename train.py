from mmengine.runner import Runner
from mmengine.config import Config

import paths


# Get default config
cfg = Config.fromfile(paths.model_config_filepath)

runner = Runner.from_cfg(cfg)
runner.train()