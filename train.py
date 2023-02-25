from mmengine.runner import Runner
from mmengine.config import Config

import paths


def main():
    # Get default config
    cfg = Config.fromfile(paths.model_config_filepath)

    runner = Runner.from_cfg(cfg)
    runner.train()

if __name__ == "__main__":
    main()
