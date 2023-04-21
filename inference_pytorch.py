import argparse
import os
import shutil
import sys
import time
from tqdm import tqdm

import mmcv
from mmdet.apis import init_detector, inference_detector
from mmyolo.utils import register_all_modules as mmyolo_register_all_modules
from mmyolo.registry import VISUALIZERS
mmyolo_register_all_modules()

repo_path = os.path.join(os.path.dirname(__file__), '..')
sys.path.append(repo_path)
import paths
from dataset import common


DEFAULT_INPUT = os.path.join(paths.proj_path, "vid", "MVI_40701.mp4")
DEFAULT_THRESHOLD = 0.3
DEFAULT_DEVICE = "cpu"


def main(args):

    video = mmcv.VideoReader(args.input)

    # Update args
    if args.number == -1:
        args.number = len(video)

    if isinstance(args.epoch, int):
        checkpoint_filepath = os.path.join(args.work_dir, f"epoch_{args.epoch}.pth")
    else:
        checkpoint_filepath = paths.get_best_checkpoint_filepath(args.work_dir)
        args.epoch = int(checkpoint_filepath.split("_")[-1].split(".")[0])

    # Get the filepath of the model configuration file
    model_config_filepath = paths.get_config_from_working_dirpath(args.work_dir)

    # Print args
    for name, value in vars(args).items():
        print(name + ":", value)
    print("input video fps:", round(video.fps, 2))
    print("model config:", model_config_filepath)
    print("checkpoint:", checkpoint_filepath)

    # More assertions
    assert os.path.exists(checkpoint_filepath), "Could not find desired checkpoint: " + checkpoint_filepath

    out_img_dirname = f"annotated_pytorch_e{args.epoch}_t{args.threshold}_" + os.path.basename(args.input).split(".")[0]
    out_img_dirpath = os.path.join(args.work_dir, out_img_dirname)
    out_vid_filename = out_img_dirname + ".mp4"
    out_vid_filepath = os.path.join(args.work_dir, out_vid_filename)
    if not os.path.exists(out_img_dirpath):
        os.mkdir(out_img_dirpath)

    # Initialize the detector
    model = init_detector(model_config_filepath, checkpoint_filepath, device=args.device)

    # Initialize a visualizer
    visualizer = VISUALIZERS.build(model.cfg.visualizer)
    visualizer.dataset_meta["classes"] = tuple(common.classes_ids.keys())

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

            # Run the inference and measure the duration
            if args.device == "cpu":
                start = time.process_time()
                result = inference_detector(model, frame)
                inference_durations.append(time.process_time() - start)
            else:
                start = time.time()
                result = inference_detector(model, frame)
                inference_durations.append(time.time() - start)

            # Visualize predictions and save to a file
            out_img_filename = str(i).zfill(6) + ".jpg"
            out_img_filepath = os.path.join(out_img_dirpath, out_img_filename)
            visualizer.add_datasample(
                name=out_img_filename,
                image=frame,
                data_sample=result,
                draw_gt=False,
                out_file=out_img_filepath,
                pred_score_thr=args.threshold)

            # Update pbar description - average inference duration
            avg_duration = sum(inference_durations) / len(inference_durations)
            if args.device == "cpu":
                pbar.set_description(f"Avg inference CPU duration: {'%.3f' % avg_duration}s")
            else:
                pbar.set_description(f"Avg inference real duration: {'%.3f' % avg_duration}s")

        print("Images annotated to", out_img_dirpath)

    except KeyboardInterrupt:
        print("KeyboardInterrupt: Stopped annotating images")

    if len(inference_durations):
        if args.device == "cpu":
            print("Average inference CPU duration:", sum(inference_durations) / len(inference_durations))
        else:
            print("Average inference real duration:", sum(inference_durations) / len(inference_durations))

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

    parser = argparse.ArgumentParser()
    parser.add_argument("work_dir",             type=str,   default=paths.working_dirpath, nargs="?",
                        help=f"working dirpath. Leave blank to use one from paths.py ({paths.working_dirpath})")
    parser.add_argument("-e", "--epoch",        type=int,
                        help="epoch number to use. Leave blank to use best based on COCO metric")
    parser.add_argument("-d", "--device",       type=str,   default=DEFAULT_DEVICE,
                        help=f"device to use for inference ('cpu', 'cuda', 'cuda:0',...). Default {DEFAULT_DEVICE}")
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
    assert os.path.exists(args.work_dir), "Working dir does not exist: " + args.work_dir
    assert args.device == "cpu" or args.device.startswith("cuda"), "Incorrect device name in --device"
    if args.device.startswith("cuda"):
        from torch.cuda import is_available
        assert is_available(), "Cuda not available on your device"
    assert os.path.exists(args.input), "Input video not found: " + args.input
    assert args.step > 0
    assert args.number >= -1
    assert 0 <= args.threshold <= 1

    main(args)
