import argparse
import cv2
import onnxruntime
import os
import mmcv
import numpy as np
import shutil
import sys
import time
from tqdm import tqdm

repo_path = os.path.join(os.path.dirname(__file__), '..')
sys.path.append(repo_path)
import paths


DEFAULT_INPUT = os.path.join(paths.proj_path, "vid", "MVI_40701.mp4")
DEFAULT_THRESHOLD = 0.3
DEFAULT_DEVICE = "cpu"

# TODO inference_common.py alebo niečo, kde bude väčšina tohto kódu...
# Môžem napríklad z onnxruntime vrátiť bboxes labels scores alebo dokonca
# PackDetInputs či ako sa volá to, čo vracia model od task_processora, a také

# TODO number of threads instead of -m flag
# TODO option to run optimized (speed-test)
# TODO option to keep original image shape (transform back to it after inference)

# TODO use MM visualizer?

def main(args):
    video = mmcv.VideoReader(args.input)

    # Update args
    if args.number == -1:
        args.number = len(video)

    # Print args
    for name, value in vars(args).items():
        print(name + ":", value)
    print("input video fps:", round(video.fps, 2))

    # More assertions
    model_filepath = os.path.join(args.work_dir, paths.deploy_onnx_filename)
    assert os.path.exists(model_filepath), f"Model was not found at {model_filepath}"

    # Paths for outputs
    out_img_dirname = f"annotated_onnxruntime_t{args.threshold}_" + os.path.basename(args.input).split(".")[0]
    out_img_dirpath = os.path.join(args.work_dir, out_img_dirname)
    out_vid_filename = out_img_dirname + ".mp4"
    out_vid_filepath = os.path.join(args.work_dir, out_vid_filename)
    if not os.path.exists(out_img_dirpath):
        os.mkdir(out_img_dirpath)

    # Initialize a onnxruntime session
    session_options = onnxruntime.SessionOptions()
    if not args.multi_thread: # Single-thread
        session_options.intra_op_num_threads = 1
        session_options.inter_op_num_threads = 1
    provider = "CPUExecutionProvider" if args.device == "cpu" else "CUDAExecutionProvider"
    session = onnxruntime.InferenceSession(model_filepath, session_options, providers=[provider])
    session.get_modelmeta()

    session_input_name = session.get_inputs()[0].name

    detector_input_shape = session.get_inputs()[0].shape[::-1][:2]
    print("Detector input shape:", detector_input_shape)

    print("Reading and annotating images")
    inference_durations = []
    try:

        pbar = tqdm(range(args.number))
        for i in pbar:

            # Read a frame
            # frame = video[i * args.step] # This doesn't work well :/
            frame_orig = video.read()
            for _ in range(args.step - 1): # This fixes it
                video.read()

            if frame_orig is None:
                break

            # Resize to detector input shape
            frame_resized = mmcv.imresize(frame_orig, detector_input_shape)

            # Pre-process
            frame = frame_resized.transpose(2, 0, 1)
            frame = frame.astype(np.float32)
            frame /= 255.
            frame = np.expand_dims(frame, axis=0)

            # Run the inference and measure the duration
            if args.device == "cpu":
                start = time.process_time()
                bboxes, labels = session.run(None, {session_input_name: frame})
                inference_durations.append(time.process_time() - start)
            else:
                start = time.time()
                bboxes, labels = session.run(None, {session_input_name: frame})
                inference_durations.append(time.time() - start)

            bboxes = np.array(bboxes[0])
            labels = np.array(labels[0])
            for j in range(len(bboxes)):
                if bboxes[j][4] < args.threshold:
                    continue

                x1, y1, x2, y2 = bboxes[j][:4].astype(int)
                confidence = round(bboxes[j][4] * 100)
                frame_resized = cv2.rectangle(frame_resized, (x1, y1), (x2, y2), (0, 255, 0), 1)
                frame_resized = cv2.putText(frame_resized, str(confidence) + ": cls=" + str(labels[j]), (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1)

            # Save the image
            out_img_filename = str(i).zfill(6) + ".jpg"
            out_img_filepath = os.path.join(out_img_dirpath, out_img_filename)
            cv2.imwrite(out_img_filepath, frame_resized)

            # Update pbar description - average inference duration
            avg_duration = sum(inference_durations) / len(inference_durations)
            if args.device == "cpu":
                pbar.set_description(f"Avg CPU duration: {'%.3f' % avg_duration}s")
            else:
                pbar.set_description(f"Avg real duration: {'%.3f' % avg_duration}s")

        print("Images annotated to", out_img_dirpath)

    except KeyboardInterrupt:
        print("KeyboardInterrupt: Stopped annotating images")

    if len(inference_durations):
        if args.device == "cpu":
            print("Average inference CPU duration:", sum(inference_durations) / len(inference_durations))
        else:
            print("Average inference real duration:", sum(inference_durations) / len(inference_durations))

    print("Converting to video")
    try:
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
    parser.add_argument("-d", "--device",       type=str,   default=DEFAULT_DEVICE,
                        help=f"device to use for inference ('cpu' or 'cuda'). Default {DEFAULT_DEVICE}")
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
    parser.add_argument("-m", "--multi-thread", action="store_true",
                        help="multi-threaded inference (when on cpu)")
    args = parser.parse_args()

    # Basic assertions
    assert os.path.exists(args.work_dir), "Working dir does not exist: " + args.work_dir
    assert args.device in ["cpu", "cuda"], "Incorrect device name in --device"
    assert os.path.exists(args.input), "Input video not found: " + args.input
    assert args.step > 0
    assert args.number >= -1
    assert 0 <= args.threshold <= 1
    if args.multi_thread and args.device == "cuda":
        print("Multi-thread mode selected but running on CUDA. Ignoring")

    main(args)
