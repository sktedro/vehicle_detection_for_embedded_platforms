"""
Runs tests (inference on testing data) on a deployed model (in ONNX format).
Settings from paths.py apply
"""
import os

# Import paths
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import paths

# Import the mmdeploy's test module
import importlib.util
spec = importlib.util.spec_from_file_location("test.py", os.path.join(paths.mmdeploy_dirpath, "tools", "test.py"))
test = importlib.util.module_from_spec(spec)
sys.modules["test.py"] = test
spec.loader.exec_module(test)

assert os.path.exists(paths.deploy_config_filepath)
assert os.path.exists(paths.model_config_filepath)

class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

args = dotdict({
        "deploy_cfg": paths.deploy_config_filepath,
        "model_cfg": paths.model_config_filepath,
        "model": [os.path.join(paths.working_dirpath, paths.deploy_onnx_filename)],
        "work_dir": paths.working_dirpath,
        "show_dir": os.path.join(paths.working_dirpath, "deployed_test/"),
        "log2file": os.path.join(paths.working_dirpath, "deployed_test_log/"),
        "device": "cpu",

        "speed_test": False,
        # "speed_test": True,

        "cfg_options": {},

        "show": False,
        "interval": 1, # Default
        "wait_time": 2, # Default
        "warmup": 10, # Default
        "log_interval": 100, # Default
        "batch_size": 1, # Default
        "uri": "192.168.1.1:60000", # Default
    })


def get_args():
    return args


def test_deployed():
    test.parse_args = get_args
    test.main()


if __name__ == "__main__":
    test_deployed()