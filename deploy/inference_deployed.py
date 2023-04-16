import onnxruntime
import mmcv
import numpy as np
import cv2
import shutil
import argparse
import time
from tqdm import tqdm

import sys, os
repo_path = os.path.join(os.path.dirname(__file__), '..')
sys.path.append(repo_path)
import paths


default_input = os.path.join(paths.proj_path, "vid", "day_hq.mp4")
default_threshold = 0.3
default_device = "cpu"

# TODO use MM visualizer?
# TODO option to keep original image shape
# TODO number of threads instead of -m flag


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("work_dir",             type=str,   default=paths.working_dirpath, nargs="?",
                        help="working dirpath. Leave blank to use one from paths.py")
    parser.add_argument("-d", "--device",         type=str,   default="cpu",
                        help="device to use for inference ('cpu' or 'cuda')")
    parser.add_argument("-s", "--step",         type=int,   default=1,
                        help="image step size (every step'th image will be taken)")
    parser.add_argument("-i", "--input",        type=str,   default=default_input,
                        help="input video file")
    parser.add_argument("-n", "--number",       type=int,   default=-1,
                        help="number of frames to annotate. -1 to annotate all")
    parser.add_argument("-t", "--threshold",    type=float, default=default_threshold,
                        help="score threshold")
    parser.add_argument("-c", "--clean",        action="store_true",
                        help="remove the images dir after finish")
    parser.add_argument("-m", "--multi-thread", action="store_true",
                        help="multi-threaded inference")
    args = parser.parse_args()

    assert os.path.exists(args.work_dir), "Working dir does not exist: " + args.work_dir

    assert os.path.exists(args.input), "Input video not found: " + args.input

    video = mmcv.VideoReader(args.input)

    if args.number == -1:
        args.number = len(video)

    print("Working dir:", args.work_dir)
    print("Input video:", args.input, "at", int(video.fps), "fps")
    print("Number of frames:", args.number, "with step", args.step)
    print("Score threshold:", args.threshold)

    out_img_dirname = f"annotated_deployed_t{args.threshold}_" + os.path.basename(args.input).split(".")[0]
    out_img_dirpath = os.path.join(args.work_dir, out_img_dirname + "/")
    out_vid_filename = out_img_dirname + ".mp4"
    out_vid_filepath = os.path.join(args.work_dir, out_vid_filename)

    if not os.path.exists(out_img_dirpath):
        os.mkdir(out_img_dirpath)

    model_filepath = os.path.join(paths.working_dirpath, paths.deploy_onnx_filename)
    assert os.path.exists(model_filepath), f"Model was not found at {model_filepath}"

    session_options = onnxruntime.SessionOptions()
    if not args.multi_thread: # Single-thread
        session_options.intra_op_num_threads = 1
        session_options.inter_op_num_threads = 1

    assert args.device in ["cpu", "cuda"], "Incorrect device name in --device"
    if args.device == "cpu":
        provider = "CPUExecutionProvider"
    else
        provider = "CUDAExecutionProvider"

    session = onnxruntime.InferenceSession(model_filepath, session_options, providers=[provider])

    session.get_modelmeta()
    first_input_name = session.get_inputs()[0].name
    first_output_name = session.get_outputs()[0].name

    detector_input_size = session.get_inputs()[0].shape[::-1][:2]
    print("Input size:", detector_input_size)

    inference_durations = []

    try:
        print("Reading and annotating images")
        pbar = tqdm(range(args.number))
        for i in pbar:

            # Get the frame, convert to rgb and run inference
            # frame = video[i * args.step] # This doesn't work well :/
            frame_orig = video.read()
            for _ in range(args.step - 1): # This fixes it
                video.read()

            if frame_orig is None:
                break

            frame_orig = mmcv.imresize(frame_orig, detector_input_size)

            frame = frame_orig.transpose(2, 0, 1)
            frame = frame.astype(np.float32)
            frame /= 255.
            frame = np.expand_dims(frame, axis=0)

            # Run the inference
            start = time.process_time()
            bboxes, labels = session.run(None, {first_input_name: frame})
            inference_durations.append(time.process_time() - start)
            bboxes = np.array(bboxes[0])
            labels = np.array(labels[0])

            out_img_filename = str(i).zfill(6) + ".jpg"
            out_img_filepath = os.path.join(out_img_dirpath, out_img_filename)

            for i in range(len(bboxes)):
                if bboxes[i][4] < args.threshold:
                    continue

                x1, y1, x2, y2 = bboxes[i][:4].astype(int)
                confidence = round(bboxes[i][4] * 100)
                frame_orig = cv2.rectangle(frame_orig, (x1, y1), (x2, y2), (0, 255, 0), 1)
                frame_orig = cv2.putText(frame_orig, str(confidence) + ": cls=" + str(labels[i]), (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1)

            cv2.imwrite(out_img_filepath, frame_orig)

            avg_cpu_time = sum(inference_durations) / len(inference_durations)
            pbar.set_description(f"Avg CPU time: {'%.3f' % avg_cpu_time}s")

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