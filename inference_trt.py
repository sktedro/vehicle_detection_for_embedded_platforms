import argparse
import os
import shutil
import time
import torch
from tqdm import tqdm

import mmcv
from mmdeploy.apis.utils import build_task_processor
from mmdeploy.utils import get_input_shape, load_config
from mmdet.datasets.transforms.loading import LoadImageFromNDArray
from mmengine.registry import TRANSFORMS
from mmyolo.registry import VISUALIZERS
from mmyolo.utils import register_all_modules as mmyolo_register_all_modules
mmyolo_register_all_modules()

import sys
repo_path = os.path.join(os.path.dirname(__file__), '..')
sys.path.append(repo_path)
import paths
import dataset.common as dataset_common
import deploy.common as deploy_common

# This is necessary because MMEngine doesn't correctly register
# LoadImageFromNDArray by itself
@TRANSFORMS.register_module()
class LoadImageFromNDArray(LoadImageFromNDArray):
    pass


DEFAULT_INPUT = os.path.join(paths.proj_path, "vid", "MVI_40701.mp4")
DEFAULT_THRESHOLD = 0.3


def main(args):
    video = mmcv.VideoReader(args.input)

    if args.number == -1:
        args.number = len(video)

    # Find deploy config
    if not os.path.exists(args.deploy_cfg):
        if not args.deploy_cfg.endswith(".py"):
            args.deploy_cfg += ".py"
        if os.path.exists(os.path.join(paths.proj_path, args.deploy_cfg)):
            args.deploy_cfg = os.path.join(paths.proj_path, args.deploy_cfg)
        if os.path.exists(os.path.join(paths.proj_path, "deploy", args.deploy_cfg)):
            args.deploy_cfg = os.path.join(paths.proj_path, "deploy", args.deploy_cfg)
    assert os.path.exists(args.deploy_cfg), f"Deploy config not found anywhere ({args.deploy_cfg})"

    # Get the filepath of the model configuration file
    model_config_filepath = paths.get_config_from_working_dirpath(args.work_dir)

    # Get model and deploy configs
    model_config = load_config(model_config_filepath)
    engine_filename = ".".join(os.path.basename(args.engine).split(".")[:-1]) # Without extension
    deploy_config = deploy_common.get_deploy_config(args.deploy_cfg,
                                                    model_config,
                                                    engine_filename)
    model_config = model_config[0]

    # Get path to the deployed engine
    if isinstance(args.engine, str):
        if not os.path.exists(args.engine):
            args.engine = os.path.join(args.work_dir, args.engine)
    else:
        engine_filename = deploy_config._cfg_dict["onnx_config"]["save_file"].split(".")[0] + ".engine"
        args.engine = os.path.join(args.work_dir, engine_filename)
    assert os.path.exists(args.engine), f"Model was not found at {args.engine}"

    # Print args
    for name, value in vars(args).items():
        print(name + ":", value)
    print("input video fps:", round(video.fps, 2))
    print("model config:", model_config_filepath)
    print("engine:", args.engine)

    out_img_dirname = f"annotated_trt_t{args.threshold}_" + os.path.basename(args.input).split(".")[0]
    out_img_dirpath = os.path.join(args.work_dir, out_img_dirname)
    out_vid_filename = out_img_dirname + ".mp4"
    out_vid_filepath = os.path.join(args.work_dir, out_vid_filename)
    if not os.path.exists(out_img_dirpath):
        os.mkdir(out_img_dirpath)

    # Initialize the detector and a visualizer
    task_processor = build_task_processor(model_config, deploy_config, "cuda")
    model = task_processor.build_backend_model([args.engine], task_processor.update_data_preprocessor)
    detector_input_shape = get_input_shape(deploy_config)
    print("Detector input shape:", detector_input_shape)

    # Initialize a visualizer
    visualizer = VISUALIZERS.build(model_config.visualizer)
    visualizer.dataset_meta["classes"] = tuple(dataset_common.classes_ids.keys())

    print("Reading and annotating images")
    inference_durations = []
    try:

        pbar = tqdm(range(args.number))
        for i in pbar:

            # Get the frame
            # frame = video[i * args.step] # This doesn't work well :/
            frame = video.read()
            for _ in range(args.step - 1): # This fixes it
                video.read()

            if frame is None:
                break

            # Pre-process
            frame = mmcv.imconvert(frame, "bgr", "rgb")
            model_inputs, _ = task_processor.create_input(frame, detector_input_shape)

            # Run the inference and measure the duration
            start = time.time()
            with torch.no_grad():
                results = model.test_step(model_inputs)
            inference_durations.append(time.time() - start)

            # Visualize predictions and save to a file
            out_img_filename = str(i).zfill(6) + ".jpg"
            out_img_filepath = os.path.join(out_img_dirpath, out_img_filename)
            visualizer.add_datasample(
                name=out_img_filename,
                image=frame,
                data_sample=results[0],
                draw_gt=False,
                out_file=out_img_filepath,
                pred_score_thr=args.threshold)

            # Update pbar description - average inference duration
            avg_duration = sum(inference_durations) / len(inference_durations)
            pbar.set_description(f"Avg inference duration: {'%.3f' % avg_duration}s")

        print("Images annotated to", out_img_dirpath)

    except KeyboardInterrupt:
        print("KeyboardInterrupt: Stopped annotating images")

    if len(inference_durations):
        print("Average inference duration:", sum(inference_durations) / len(inference_durations))

    try:
        print("Converting to video")
        mmcv.frames2video(
            out_img_dirpath,
            out_vid_filepath,
            fps=int(video.fps // args.step),
            fourcc="mp4v")
        print("Video saved to", out_vid_filepath)

    except KeyboardInterrupt:
        print("KeyboardInterrupt: Stopped converting to video")

    if args.clean:
        shutil.rmtree(out_img_dirpath)
        print("Removed", out_img_dirpath)


if __name__ == "__main__":

    # TODO remove
    # deploy_cfg = "/home/user/proj/deploy/detection_tensorrt_static-640x640.py"

    # TODO engine filename from deploy cfg?
    parser = argparse.ArgumentParser()
    parser.add_argument("deploy_cfg",           type=str,
                        help="Deploy config filepath. Can be relative to deploy/ folder or relative to the project folder")
    parser.add_argument("work_dir",             type=str,   default=paths.working_dirpath, nargs="?",
                        help=f"working dirpath. Leave blank to use one from paths.py ({paths.working_dirpath})")
    parser.add_argument("-e", "--engine",       type=str,
                        help="inference engine filepath or filename. Defaultly taken from deploy configuration file")
    parser.add_argument("-i", "--input",        type=str,   default=DEFAULT_INPUT,
                        help=f"input video file. Default {DEFAULT_INPUT}")
    parser.add_argument("-s", "--step",         type=int,   default=1,
                        help="image step size (every step'th image will be taken). Default 1")
    parser.add_argument("-n", "--number",       type=int,   default=-1,
                        help="number of frames to annotate. Default -1 to annotate all")
    parser.add_argument("-t", "--threshold",    type=float, default=DEFAULT_THRESHOLD,
                        help=f"score threshold. Default {DEFAULT_THRESHOLD}")
    parser.add_argument("-c", "--clean",        action="store_true",
                        help="remove the images dir after finish")
    args = parser.parse_args()

    # Basic assertions
    assert torch.cuda.is_available(), "Cuda not available on your device"
    assert os.path.exists(args.work_dir), "Working dir does not exist: " + args.work_dir
    assert os.path.exists(args.input), "Input video not found: " + args.input
    assert args.step > 0
    assert args.number >= -1
    assert 0 <= args.threshold <= 1

    main(args)