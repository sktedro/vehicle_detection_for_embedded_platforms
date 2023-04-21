# TODO update based on inference_onnxruntime.py - keď už bude hotovo
import os
import argparse
import shutil
import time
from tqdm import tqdm
import mmcv
import numpy as np
import torch
import mmcv

import sys
repo_path = os.path.join(os.path.dirname(__file__), '..')
sys.path.append(repo_path)
import paths
from dataset import common

from mmdeploy.apis.utils import build_task_processor

from mmdeploy.utils import get_input_shape, load_config

# MMYOLO integration
from mmyolo.utils import register_all_modules
from mmyolo.registry import VISUALIZERS
register_all_modules()

from mmengine.registry import TRANSFORMS

from mmyolo.utils import register_all_modules as mmyolo_reg
mmyolo_reg()

from mmdet.utils import register_all_modules as mmdet_reg
from mmdet.datasets.transforms import loading
a = loading.LoadImageFromNDArray()
mmdet_reg()

@TRANSFORMS.register_module()
class LoadImageFromNDArray(loading.LoadImageFromFile):
    """Load an image from ``results['img']``.

    Similar with :obj:`LoadImageFromFile`, but the image has been loaded as
    :obj:`np.ndarray` in ``results['img']``. Can be used when loading image
    from webcam.

    Required Keys:

    - img

    Modified Keys:

    - img
    - img_path
    - img_shape
    - ori_shape

    Args:
        to_float32 (bool): Whether to convert the loaded image to a float32
            numpy array. If set to False, the loaded image is an uint8 array.
            Defaults to False.
    """

    def transform(self, results: dict) -> dict:
        """Transform function to add image meta information.

        Args:
            results (dict): Result dict with Webcam read image in
                ``results['img']``.

        Returns:
            dict: The dict contains loaded image and meta information.
        """

        img = results['img']
        if self.to_float32:
            img = img.astype(np.float32)

        results['img_path'] = None
        results['img'] = img
        results['img_shape'] = img.shape[:2]
        results['ori_shape'] = img.shape[:2]
        return results



default_input = os.path.join(paths.proj_path, "vid", "MVI_40701.mp4")
default_threshold = 0.3


if __name__ == "__main__":

    # TODO:
    deploy_cfg = "/home/user/bp/proj/deploy/detection_tensorrt_static-640x640.py"

    # TODO:
    engine = "/home/user/bp/proj/working_dir_yolov8_m_384_conf8/end2end.engine"

    from torch.cuda import is_available
    assert is_available(), "Cuda not available on your device"

    parser = argparse.ArgumentParser()
    parser.add_argument("work_dir",          type=str,   default=paths.working_dirpath, nargs="?",
                        help="working dirpath. Leave blank to use one from paths.py")
    parser.add_argument("-s", "--step",      type=int,   default=1,
                        help="image step size (every step'th image will be taken)")
    parser.add_argument("-i", "--input",     type=str,   default=default_input,
                        help="input video file")
    parser.add_argument("-n", "--number",    type=int,   default=-1,
                        help="number of frames to annotate. -1 to annotate all")
    parser.add_argument("-d", "--device",    type=str,   default="cuda:0",
                        help="device for inference, default cuda:0")
    parser.add_argument("-t", "--threshold", type=float, default=default_threshold,
                        help="score threshold")
    parser.add_argument("-c", "--clean",     action="store_true",
                        help="remove the images dir after finish")
    args = parser.parse_args()

    assert os.path.exists(args.work_dir), "Working dir does not exist: " + args.work_dir

    # Get the filepath of the model configuration file (should be the only file
    # in the work dir with .py extension
    model_config_filepath = [f for f in os.listdir(args.work_dir) if f.split(".")[-1] == "py"]
    assert len(model_config_filepath) == 1, "Could not find model config in the working dir or there are more than one python file: " + str(model_config_filepath)
    model_config_filepath = os.path.join(args.work_dir, model_config_filepath[0])

    assert os.path.exists(args.input), "Input video not found: " + args.input


    video = mmcv.VideoReader(args.input)

    if args.number == -1:
        args.number = len(video)

    deploy_cfg, model_cfg = load_config(deploy_cfg, model_config_filepath)
    task_processor = build_task_processor(model_cfg, deploy_cfg, args.device)
    model = task_processor.build_backend_model([engine], task_processor.update_data_preprocessor)
    # model = task_processor.init_backend_model([engine])
    input_shape = get_input_shape(deploy_cfg)

    print("Working dir:", args.work_dir)
    print("Model:", model_config_filepath)
    print("Input video:", args.input, "at", int(video.fps), "fps")
    print("Device:", args.device)
    print("Number of frames:", args.number, "with step", args.step)
    print("Score threshold:", args.threshold)

    out_img_dirname = f"annotated_trt_t{args.threshold}_" + os.path.basename(args.input).split(".")[0]
    out_img_dirpath = os.path.join(args.work_dir, out_img_dirname + "/")
    out_vid_filename = out_img_dirname + ".mp4"
    out_vid_filepath = os.path.join(args.work_dir, out_vid_filename)

    visualizer = VISUALIZERS.build(model_cfg.visualizer)
    visualizer.dataset_meta["classes"] = tuple(common.classes_ids.keys())

    if not os.path.exists(out_img_dirpath):
        os.mkdir(out_img_dirpath)

    try:
        inference_durations = []

        print("Reading and annotating images")
        for i in tqdm(range(args.number)):
            # Get the frame, convert to rgb and run inference
            # frame = video[i * args.step] # This doesn't work well :/
            frame = video.read()
            for _ in range(args.step - 1): # This fixes it
                video.read()

            if frame is None:
                break

            frame = mmcv.imconvert(frame, "bgr", "rgb")

            model_inputs, _ = task_processor.create_input(frame, input_shape)

            start = time.process_time()
            start_real = time.time()
            with torch.no_grad():
                # result = task_processor.run_inference(model, model_inputs)
                result = model.test_step(model_inputs)
            print(time.time() - start_real)
            inference_durations.append(time.process_time() - start)

            # Visualize predictions and save to a file
            out_img_filename = str(i).zfill(6) + ".jpg"
            out_img_filepath = os.path.join(out_img_dirpath, out_img_filename)
            visualizer.add_datasample(
                name=out_img_filename,
                image=frame,
                data_sample=result[0],
                draw_gt=False,
                out_file=out_img_filepath,
                pred_score_thr=args.threshold)

            # Without mmyolo, something like this worked:
            # if score_thr == 0:
            #     model.show_result(frame, result, out_file=out_filepath)
            # else:
            #     model.show_result(frame, result, score_thr=score_thr, out_file=out_filepath)

        print("Images annotated")

    except KeyboardInterrupt:
        print("KeyboardInterrupt: Stopped annotating images")

    if len(inference_durations):
        print("Average inference CPU time:", sum(inference_durations) / len(inference_durations))

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