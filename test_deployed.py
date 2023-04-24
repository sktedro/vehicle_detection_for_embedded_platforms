"""
Runs tests (inference on testing data) on a deployed model.
"""
import argparse
import os

# Import paths
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import paths
from dataset import common as dataset_common

# Import the mmdeploy's test.py module
import importlib.util
spec = importlib.util.spec_from_file_location("test.py", os.path.join(paths.mmdeploy_dirpath, "tools", "test.py"))
test = importlib.util.module_from_spec(spec)
sys.modules["test.py"] = test
spec.loader.exec_module(test)

# Acts like a dictionary of arguments - to pass to the mmdeploy's test.py module
class dotdict(dict):
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


DEFAULT_TEST_DATASET = os.path.join(
    paths.datasets_dirpath, 
    dataset_common.gt_combined_filenames["test"])


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("deploy_cfg",           type=str,
                        help="Deploy config filepath. Can be relative to deploy/ folder or relative to the project folder")
    parser.add_argument("work_dir",             type=str,
                        help="working dirpath. Preferably relative to the proj path or absolute")
    parser.add_argument("-f", "--filepath",     type=str,
                        help="model filepath, can be relative to the working dir, eg. end2end.onnx")
    parser.add_argument("-d", '--device',       type=str,
                        help="device used for inference, cpu or cuda")
    # TODO allow for testing with custom dataset
    # parser.add_argument("-t", "--test-dataset", type=str, default=DEFAULT_TEST_DATASET,
    #                     help=f"COCO dataset filepath to test with. Default {DEFAULT_TEST_DATASET}")
    parser.add_argument("-b", "--batch-size",   type=int, default=1,
                        help="test batch size. Default 1")
    parser.add_argument("-i", "--interval",     type=int, default=1,
                        help="only every i-th sample will be visualized to a file. Default 1")
    parser.add_argument("-w", "--warmup",       type=int, default=10,
                        help="warmup during speed test. Speed is not calculated for the first w samples. Default 10")
    parser.add_argument("-s", "--speed-test",   action="store_true",
                        help="test and print inference speed")
    args = parser.parse_args()

    return args


def main(args):
    if os.path.exists(os.path.join(paths.proj_path, args.work_dir)):
        args.work_dir = os.path.join(paths.proj_path, args.work_dir)

    model_cfg = paths.get_config_from_working_dirpath(args.work_dir)

    if os.path.exists(args.filepath):
        model_filepath = args.filepath
    elif os.path.exists(os.path.join(args.work_dir, args.filepath)):
        model_filepath = os.path.join(args.work_dir, args.filepath)
    else:
        raise Exception(f"Model file not found ({args.filepath})")

    model_name = os.path.basename(model_filepath).replace(".", "_")

    out_dirpath = os.path.join(args.work_dir, "test_" + model_name)
    log_dirpath = os.path.join(args.work_dir, "test_" + model_name + "_log")

    assert os.path.exists(args.deploy_cfg), f"Deploy config not found at {args.deploy_cfg}"
    assert os.path.exists(args.work_dir), f"Working dir not found at {args.work_dir}"
    assert args.device == "cpu" or args.device.startswith("cuda")
    assert args.interval >= 0
    assert args.warmup >= 0
    assert args.batch_size > 0

    # Build a dotdict for the test.py script to act as args
    # All args need to be here - argparse returns default if argument is not
    # given and we are acting like argparse
    mmdeploy_args = dotdict({
            "deploy_cfg": args.deploy_cfg,
            "work_dir": args.work_dir,
            "model_cfg": model_cfg,
            "model": [model_filepath],
            "show_dir": out_dirpath,
            "log2file": log_dirpath,
            "device": args.device,
            "speed_test": args.speed_test,
            "cfg_options": {},
            "show": False,
            "wait_time": 2, # Default 2
            "interval": args.interval,
            "warmup": args.warmup,
            "log_interval": 100, # Default
            "batch_size": args.batch_size,
            "uri": "192.168.1.1:60000", # Default
        })

    # Replace the test.py's parse_args function with a new one
    def get_args():
        return mmdeploy_args
    test.parse_args = get_args

    # Run the test
    test.main()

if __name__ == "__main__":
    main(parse_args())