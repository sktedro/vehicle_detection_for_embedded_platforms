
"""
Deploys all available working dirs with all possible configs, automatically

To change which epoch is deployed, simply put `epoch_to_deploy` file to the
working dir containing the epoch number (eg. '30')
"""
import argparse
import logging
import os
import sys
import time
from datetime import datetime
from pprint import pformat
from concurrent.futures import ThreadPoolExecutor

if __name__ == "__main__":
    import deploy_model
else:
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    import deploy_model

repo_path = os.path.join(os.path.dirname(__file__), '..')
sys.path.append(repo_path)
import paths


# Acts like a dictionary of arguments - to pass to the mmdeploy's test.py module
class dotdict(dict):
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


def run_task(output_filepath, args, logger):
    logger.info("Task begin: " + output_filepath)
    logger.info(pformat(args))
    deploy_model.main(args)
    logger.info("Task done: " + output_filepath)


def get_all_tasks(args, logger):
    # Get a list of all work dirs
    work_dirpaths = []
    for file in os.listdir(paths.proj_path):
        filepath = os.path.join(paths.proj_path, file)
        if os.path.isdir(filepath):
            if os.path.basename(filepath).startswith("work"):
                work_dirpaths.append(filepath)
    work_dirpaths.sort()

    logger.info(f"{len(work_dirpaths)} working dirs found to use")

    # Get a list of all deploy configs
    deploy_config_filepaths = []
    for file in os.listdir(os.path.join(paths.proj_path, "deploy")):
        filepath = os.path.join(paths.proj_path, "deploy", file)
        if os.path.basename(filepath).startswith("config_") and filepath.endswith(".py"):
            if args.backend is not None and args.backend not in file:
                continue
            if "onnxruntime" in file or "tensorrt" in file:
                deploy_config_filepaths.append(filepath)
            else:
                print(f"Config file {filepath} ignored because it is for an unknown engine")
    deploy_config_filepaths.sort()

    logger.info(f"{len(deploy_config_filepaths)} deploy configs found to use")

    # Create a dictionary containing all combinations to deploy
    tasks = { # Examples:
        "<working_dirname>": {
            "working_dirpath": ".../<working_dirname>",
            "checkpoint_filepath": ".../<working_dirname>/epoch_500.pth",
            "checkpoint_epoch_number": 500,
            "configs": [
                {
                    "config_filepath": ".../deploy/config_onnxruntime_static.py",
                    "backend": "onnxruntime",
                    "config_name": "onnxruntime_static",
                    "output_filename": "onnxruntime_static.onnx",
                    "output_filepath": ".../<working_dirname>/onnxruntime_static.onnx",
                }
            ]
        }
    }
    tasks = {}
    for working_dirpath in work_dirpaths:
        working_dirname = os.path.basename(working_dirpath)
        tasks[working_dirname] = {}
        tasks[working_dirname]["working_dirpath"] = working_dirpath
        tasks[working_dirname]["configs"] = []

        # Get checkpoint (epoch number and the checkpoint filepath)
        if os.path.exists(os.path.join(working_dirpath, "epoch_to_deploy")):
            with open(os.path.join(working_dirpath, "epoch_to_deploy")) as f:
                epoch_number = int(f.read())
                tasks[working_dirname]["checkpoint_epoch_number"] = epoch_number
                tasks[working_dirname]["checkpoint_filepath"] = os.path.join(
                    working_dirpath, "epoch_" + str(epoch_number) + ".pth")
        else:
            checkpoint_filepath = paths.get_best_checkpoint_filepath(working_dirpath)
            epoch_number, _ = os.path.splitext(checkpoint_filepath.split("_")[-1])
            tasks[working_dirname]["checkpoint_epoch_number"] = int(epoch_number)
            tasks[working_dirname]["checkpoint_filepath"] = checkpoint_filepath

        # Get deploy configs
        for deploy_config_filepath in deploy_config_filepaths:
            deploy_config_filename = os.path.basename(deploy_config_filepath)

            backend = deploy_config_filename.split("_")[1] # config_onnxruntime_static -> onnxruntime
            config_name, _ = os.path.splitext( # config_onnxruntime.py -> onnxruntime
                "_".join(deploy_config_filename.split("_")[1:]))
            if backend == "tensorrt":
                output_filename = config_name + ".engine" # static.engine
            elif backend == "onnxruntime":
                output_filename = config_name + ".onnx" # static.onnx
            output_filepath = os.path.join(working_dirpath, output_filename) # .../work_dir/static.onnx

            tasks[working_dirname]["configs"].append({
                "config_filepath": deploy_config_filepath,
                "backend": backend,
                "config_name": config_name,
                "output_filename": output_filename,
                "output_filepath": output_filepath,
            })

    return tasks


def main(args):

    print("If you also want to log everything the subprocesses output, please use bash redirection to your own file. Logger here only logs output from this file")
    time.sleep(1)

    general_log_filepath = os.path.join(
        paths.proj_path,
        "deploy_all_log_" + datetime.today().strftime('%Y-%m-%d_%H-%M-%S') + ".txt")
    print("Logging to:", general_log_filepath)

    # Log to stdout and a file
    logFormatter = logging.Formatter("%(asctime)s: %(message)s")
    logger = logging.getLogger("deploy_all.py")
    logger.setLevel(logging.DEBUG)
    consoleHandler = logging.StreamHandler(sys.stdout)
    consoleHandler.setFormatter(logFormatter)
    logger.addHandler(consoleHandler)
    consoleHandler = logging.FileHandler(general_log_filepath)
    consoleHandler.setFormatter(logFormatter)
    logger.addHandler(consoleHandler)

    assert os.path.exists(paths.proj_path)

    tasks = get_all_tasks(args, logger)

    logger.info("Tasks:")
    logger.info(pformat(tasks))
    logger.info(f"Total: {sum(len(task['configs']) for task in tasks)} tasks")

    with ThreadPoolExecutor(args.deploy_jobs) as executor:
        futures = []
        for working_dirname in tasks:
            for config in tasks[working_dirname]["configs"]:

                # Check if the file already exists and maybe skip it
                if os.path.exists(config["output_filepath"]):
                    if args.replace_existing:
                        logger.info(f"{config['output_filepath']} exists. Replacing")
                    else:
                        logger.info(f"{config['output_filepath']} exists. Skipping")
                        continue

                device = "cuda" if config["backend"] == "tensorrt" else "cpu"
                if "best_coco" in tasks[working_dirname]["checkpoint_filepath"]:
                    epoch = None
                else:
                    epoch = tasks[working_dirname]["checkpoint_epoch_number"]

                deploy_args = dotdict({
                        "deploy_cfg": config["config_filepath"],
                        "work_dir": tasks[working_dirname]["working_dirpath"],
                        "epoch": epoch,
                        "output_filename": config["config_name"], # deploy_model doesn't want the extension
                        "device": device,
                        "visualize": False,
                        "log_level": "WARNING",
                    })
                futures.append(executor.submit(run_task, config["output_filepath"], deploy_args, logger))

        try:
            for i in range(len(futures)):
                result = futures[i].result()
                logger.info(f"Future number {i + 1} of {len(futures)} done")
                logger.info(result)
        except KeyboardInterrupt:
            logger.error("KeyboardInterrupt. Exiting")
            for future in futures:
                try:
                    future.cancel()
                except:
                    pass

    logger.info("All done")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-j", "--deploy-jobs",      type=int, default=1,
                        help="run deployment jobs in parallel. Default 1. Large numbers may result in RAM getting full")
    parser.add_argument("-b", "--backend",          type=str,
                        help="deploy only to one backend ('onnxruntime' or 'tensorrt')")
    parser.add_argument("-r", "--replace-existing", action="store_true",
                        help="whether to replace existing models if found")
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
