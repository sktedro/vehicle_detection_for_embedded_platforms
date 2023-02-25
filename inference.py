import os
import argparse
import shutil
from tqdm import tqdm

import mmcv
from mmdet.apis import init_detector, inference_detector

# MMYOLO integration
from mmyolo.utils import register_all_modules
from mmyolo.registry import VISUALIZERS
register_all_modules()

import paths
from dataset import common


default_input = "day_hq.mp4"
default_threshold = 0.3
default_device = "cpu"


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("work_dir",          type=str,
                        help="working dirpath. Leave blank to use one from paths.py")
    parser.add_argument("-e", "--epoch",     type=int,
                        help="epoch number to use. Leave blank to use latest")
    parser.add_argument("-s", "--step",      type=int,   default=1,
                        help="image step size (every step'th image will be taken)")
    parser.add_argument("-i", "--input",     type=str,   default=default_input,
                        help="input video file")
    parser.add_argument("-n", "--number",    type=int,   default=-1,
                        help="number of frames to annotate. -1 to annotate all")
    parser.add_argument("-d", "--device",    type=str,   default=default_device,
                        help="device for inference, cpu or cuda")
    parser.add_argument("-t", "--threshold", type=float, default=default_threshold,
                        help="score threshold")
    parser.add_argument("-c", "--clean",     action="store_true",
                        help="remove the images dir after finish")
    args = parser.parse_args()

    assert isinstance(args.epoch, int) or paths.last_checkpoint_filepath != None, "Epoch number not given and last checkpoint filepath empty"

    # If epoch number was provided, update the checkpoint filepath
    if isinstance(args.epoch, int):
        paths.last_checkpoint_filepath = os.path.join(paths.working_dirpath, f"epoch_{args.epoch}.pth")
    # Else, update the epoch number
    else:
        args.epoch = int(paths.last_checkpoint_filepath.split("_")[-1].split(".")[0])
    assert os.path.exists(paths.last_checkpoint_filepath), "Could not find desired checkpoint: " + paths.last_checkpoint_filepath

    # If work dir was not provided, read it from paths.py
    if not isinstance(args.work_dir, str):
        args.work_dir = paths.working_dirpath
    assert os.path.exists(args.work_dir), "Working dir does not exist: " + args.work_dir

    # Get the filepath of the model configuration file (should be the only file
    # in the work dir with .py extension
    model_config_filepath = [f for f in os.listdir(args.work_dir) if f.split(".")[-1] == "py"]
    assert len(model_config_filepath) == 1, "Could not find model config in the working dir or there are more than one python file: " + model_config_filepath
    model_config_filepath = os.path.join(args.work_dir, model_config_filepath[0])

    assert os.path.exists(args.input), "Input video not found: " + args.input

    if args.device.startswith("cuda"):
        from torch.cuda import is_available
        assert is_available(), "Cuda not available on your device"

    video = mmcv.VideoReader(args.input)

    if args.number == -1:
        args.number = len(video)

    print("Working dir:", args.work_dir)
    print("Model:", paths.model_config_filepath)
    print("Checkpoint:", paths.last_checkpoint_filepath)
    print("Input video:", args.input, "at", int(video.fps), "fps")
    print("Device:", args.device)
    print("Number of frames:", args.number, "with step", args.step)
    print("Score threshold:", args.threshold)

    out_img_dirname = os.path.basename(args.input).split(".")[0] + f"_e{args.epoch}_{args.threshold}"
    out_img_dirpath = os.path.join(args.work_dir, out_img_dirname + "/")
    out_vid_filename = out_img_dirname + ".mp4"
    out_vid_filepath = os.path.join(args.work_dir, out_vid_filename)

    model = init_detector(model_config_filepath, paths.last_checkpoint_filepath, device=args.device)
    visualizer = VISUALIZERS.build(model.cfg.visualizer)
    visualizer.dataset_meta["classes"] = tuple(common.classes_ids.keys())

    if not os.path.exists(out_img_dirpath):
        os.mkdir(out_img_dirpath)

    try:
        print("Reading and annotating images")
        for i in tqdm(range(args.number)):

            frame = video[i * args.step]
            frame = mmcv.imconvert(frame, "bgr", "rgb")
            result = inference_detector(model, frame)

            # Visualize predictions and save to a file
            out_img_filepath = os.path.join(out_img_dirpath, str(i).zfill(6) + ".jpg")
            visualizer.add_datasample(
                name=os.path.basename(out_img_filepath),
                image=frame,
                data_sample=result,
                draw_gt=False,
                out_file=out_img_filepath,
                pred_score_thr=args.threshold)

            # Without mmyolo, something like this worked:
            # if score_thr == 0:
            #     model.show_result(frame, result, out_file=out_filepath)
            # else:
            #     model.show_result(frame, result, score_thr=score_thr, out_file=out_filepath)

            if i == args.number: # Doesn't stop if stop_at_frames == 0
                break
        print("Images annotated")

    except KeyboardInterrupt:
        print("KeyboardInterrupt: Stopped annotating images")

    try:
        print("Converting to video")
        mmcv.frames2video(
            out_img_dirpath,
            out_vid_filepath,
            fps=video.fps // args.step,
            fourcc="mp4v")
        print("Video saved to", out_vid_filepath)
    except KeyboardInterrupt:
        print("KeyboardInterrupt: Stopped converting to video")

    if args.clean:
        shutil.rmtree(out_img_dirpath)
        print("Removed", out_img_dirpath)