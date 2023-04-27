"""
Runs tests on all available working dirs with all possible configs,
automatically
"""
import argparse
import logging
import os
import subprocess
import sys
import time
from datetime import datetime
from pprint import pformat

import test_deployed
from deploy import deploy_all

repo_path = os.path.join(os.path.dirname(__file__), '..')
sys.path.append(repo_path)
import paths


# Acts like a dictionary of arguments - to pass to the mmdeploy's test.py module
class dotdict(dict):
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


def run_task(args, logger):
    logger.info("Task begin: " + args["filepath"])
    logger.info(pformat(args))

    def make_arg(arg):
        """ work_dir -> --work-dir """
        return "--" + arg.replace("_", "-")

    cmd = ["python3"]
    cmd += [os.path.join(os.path.abspath(os.path.dirname(__file__)), 'test_deployed.py')]
    cmd += [args["deploy_cfg"]]
    cmd += [args["work_dir"]]
    cmd += [args["filepath"]]
    cmd += [args["device"]]
    for key in args:
        if key in ["deploy_cfg", "work_dir", "filepath", "device"]:
            continue
        if isinstance(args[key], bool):
            if args[key] == True:
                cmd += [make_arg(key)]
        elif isinstance(args[key], str):
            cmd += [make_arg(key), args[key]]
        else:
            cmd += [make_arg(key), str(args[key])]

    logger.info("Executing: " + str(cmd))
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    with p.stdout:
        for line in iter(p.stdout.readline, b''):
            logger.debug(line.decode("utf-8").strip())
    logger.info("Executed: " + str(cmd))

    # p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    # logger.info("Executed: " + str(cmd))
    # logger.debug("Return code: " + str(p.returncode))
    # logger.debug("Output: " + p.stdout.decode())

    logger.info("Task done: " + args["filepath"])


def get_all_tasks(args, logger):
    # We can use get_all_tasks() from deploy_all.py
    deploy_all_args = dotdict({
        "deploy_jobs": 1,
        "backend": args.backend,
        "replace_existing": True
        })
    deploy_tasks = deploy_all.get_all_tasks(deploy_all_args, logger)

    tasks = [ # Examples:
        {
            "deploy_cfg": ".../deploy/config_onnxruntime_static.py",
            "work_dir": ".../<working_dirname>",
            "filepath": "onnxruntime_static.onnx",
            "device": "cuda",
            "batch_size": 4,
            "interval": 0,
            "warmup": 10,
            "speed_test": True,
        }
    ]
    tasks = []
    for working_dirname in deploy_tasks:
        for config in deploy_tasks[working_dirname]["configs"]:
            for batch_size in args.batch_sizes:
                if batch_size != 1 and "dynamic" not in os.path.basename(config["config_filepath"]):
                    continue
                tasks.append({
                    "deploy_cfg": config["config_filepath"],
                    "work_dir": deploy_tasks[working_dirname]["working_dirpath"],
                    "filepath": config["output_filename"],
                    "device": args.device,
                    "batch_size": batch_size,
                    "visualize": args.visualize,
                    "warmup": args.warmup,
                    "speed_test": True,
                    })

    return tasks

def filter_tasks(args, tasks, logger):
    # Filter to existing tasks
    for task in tasks.copy():
        model_filepath = os.path.join(task["work_dir"], task["filepath"])
        if not os.path.exists(model_filepath):
            logger.warning(f"Model at {model_filepath} does not exist! Skipping")
            tasks.remove(task)

    # Filter out already done tasks
    if not args.replace_existing:
        for task in tasks.copy():
            log_filepath = get_logfile(task)
            if os.path.exists(log_filepath):
                logger.warning(f"Test log at {log_filepath} found! Skipping")
                tasks.remove(task)

    return tasks


def get_tasks(args, logger):
    all_tasks = get_all_tasks(args, logger)
    return filter_tasks(args, all_tasks, logger)


def get_logfile(task):
    return test_deployed.get_log_file(
        task["work_dir"],
        os.path.basename(task["filepath"]).replace(".", "_"),
        task["batch_size"])


def main(args):
    print("If you also want to log everything the subprocesses output, please use bash redirection to your own file. Logger here only logs output from this file")
    time.sleep(1)

    general_log_filepath = os.path.join(
        paths.proj_path,
        "test_all_log_" + datetime.today().strftime('%Y-%m-%d_%H-%M-%S') + ".txt")
    print("Logging to:", general_log_filepath)

    # Log to stdout and a file
    logFormatter = logging.Formatter("%(asctime)s: %(message)s")
    logger = logging.getLogger("test_all.py")
    logger.setLevel(logging.DEBUG)
    consoleHandler = logging.StreamHandler(sys.stdout)
    consoleHandler.setFormatter(logFormatter)
    logger.addHandler(consoleHandler)
    consoleHandler = logging.FileHandler(general_log_filepath)
    consoleHandler.setFormatter(logFormatter)
    logger.addHandler(consoleHandler)

    assert os.path.exists(paths.proj_path)

    args.batch_sizes = list(set(args.batch_sizes)) # Make them unique
    args.batch_sizes.sort()

    assert args.device in ["cpu", "cuda"], f"Unknown device: {args.device}"

    if args.backend is not None:
        assert args.backend in ["onnxruntime", "tensorrt"], f"Unknown backend: {args.backend}"

    if args.backend == "tensorrt" and args.device == "cpu":
        logger.warning("Test on CPU requested with TensorRT backend. Ignoring and using CUDA")
        args.device = "cuda"

    tasks = get_tasks(args, logger)

    logger.info("Tasks:")
    logger.info(pformat(tasks))
    logger.info(f"Total: {len(tasks)} tasks")

    # Run all tasks
    for i in range(len(tasks)):
        args = dotdict(tasks[i])
        try:
            run_task(args, logger)
            logger.info(f"Task number {i + 1} of {len(tasks)} done")
        except KeyboardInterrupt:
            log_filepath = get_logfile(tasks[i])
            logger.error(f"Task interrupted. Deleting log file at {log_filepath} and exiting")
            os.remove(log_filepath)
            logger.info(f"Log file at {log_filepath} removed")
            break


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-b", "--backend",          type=str,
                        help="test only files of one backend ('onnxruntime' or 'tensorrt')")
    parser.add_argument("-d", "--device",           type=str, default="cpu",
                        help="device to use for testing ('cpu' or 'cuda') if using onnxruntime backend. Default 'cpu'")
    parser.add_argument("-r", "--replace-existing", action="store_true",
                        help="whether to test even if existing results are found (and overwrite them)")
    parser.add_argument("-s", "--batch-sizes",      type=int, action="append", default=[1],
                        help="testing batch size(s). Only applicable to dynamic models. Use more with '-b 1 -b 2 -b 4'. Default 1")
    parser.add_argument("-v", "--visualize",        action="store_true",
                        help="visualize the test ouput to a directory (in the work dir)")
    parser.add_argument("-w", "--warmup",           type=int, default=10,
                        help="warmup during speed test. Speed is not calculated for the first w samples. Default 10")
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
