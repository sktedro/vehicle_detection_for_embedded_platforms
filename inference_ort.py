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
    # Basic assertions
    assert os.path.exists(args.work_dir), "Working dir does not exist: " + args.work_dir
    assert args.device in ["cpu", "cuda"], "Incorrect device name in --device"
    assert os.path.exists(args.input), "Input video not found: " + args.input
    assert args.step >= 1
    assert args.batch_size >= 1
    assert args.number >= -1
    assert 0 <= args.threshold <= 1
    if args.multi_thread and args.device == "cuda":
        print("Multi-thread mode selected but running on CUDA. Ignoring")

    video = mmcv.VideoReader(args.input)

    # Update args
    if args.number == -1:
        args.number = len(video)

    model_filepath = os.path.join(args.work_dir, paths.deploy_onnx_filename)

    # Print args
    for name, value in vars(args).items():
        print(name + ":", value)
    print("input video fps:", round(video.fps, 2))
    print("model:", model_filepath)

    # More assertions
    assert os.path.exists(model_filepath), f"Model was not found at {model_filepath}"

    # Paths for outputs
    out_img_dirname = f"annotated_onnxruntime_t{args.threshold}_" + os.path.basename(args.input).split(".")[0]
    out_img_dirpath = os.path.join(args.work_dir, out_img_dirname)
    out_vid_filename = out_img_dirname + ".mp4"
    out_vid_filepath = os.path.join(args.work_dir, out_vid_filename)
    if not os.path.exists(out_img_dirpath):
        os.mkdir(out_img_dirpath)

    # Initialize an onnxruntime session
    session_options = onnxruntime.SessionOptions()
    if not args.multi_thread: # Single-thread
        session_options.intra_op_num_threads = 1
        session_options.inter_op_num_threads = 1
    provider = "CPUExecutionProvider" if args.device == "cpu" else "CUDAExecutionProvider"
    session = onnxruntime.InferenceSession(model_filepath, session_options, providers=[provider])
    session.get_modelmeta()
    session_input_name = session.get_inputs()[0].name
    input_h, input_w = session.get_inputs()[0].shape[2:] # It should be h*w
    dynamic_batch_size = session.get_inputs()[0].shape[0] != 1
    assert dynamic_batch_size or args.batch_size == 1, "Batch size needs to be 1 for static models"
    print("Detector input shape (w, h):", input_w, input_h)
    print("Allow dynamic batch size:", dynamic_batch_size)

    print("Reading and annotating images")
    inference_durations = []
    try:

        pbar = tqdm(range(args.number))
        for i in range(0, args.number, args.batch_size): # Progress bar will be updated manually based on the batch size

            frames_orig = []
            frames_resized = []
            frames_preprocessed = []

            # Read a frame
            for _ in range(args.batch_size):

                # This is necessary - indexing in the video doesn't work well
                frame = video.read()
                for _ in range(args.step - 1): # Skip frames according to step
                    video.read()

                if frame is None:
                    break

                frames_orig.append(frame)

                # Resize to detector input shape
                frame = mmcv.imresize(frame, (input_w, input_h))
                frames_resized.append(frame)

                # Pre-process - channels = channels,h,w
                frame = frame.transpose(2, 0, 1).astype(np.float32) / 255.
                frames_preprocessed.append(frame)

            if len(frames_orig) == 0:
                break

            detector_input = np.array(frames_preprocessed)

            # Run the inference and measure the duration
            if args.device == "cpu":
                start = time.process_time()
                bboxes, labels = session.run(None, {session_input_name: detector_input})
                inference_durations.append(time.process_time() - start)
            else:
                start = time.time()
                bboxes, labels = session.run(None, {session_input_name: detector_input})
                inference_durations.append(time.time() - start)

            for img_in_batch_i in range(len(frames_resized)):
                frame = frames_resized[img_in_batch_i]
                for detection_i in range(len(bboxes[img_in_batch_i])):
                    if bboxes[img_in_batch_i][detection_i][4] < args.threshold:
                        continue

                    x1, y1, x2, y2 = bboxes[img_in_batch_i][detection_i][:4].astype(int)
                    confidence = round(bboxes[img_in_batch_i][detection_i][4] * 100)
                    frame = cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 1)
                    frame = cv2.putText(frame, str(confidence) + ": cls=" + str(labels[img_in_batch_i][detection_i]), (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1)

                # Save the image
                img_number = i + img_in_batch_i
                out_img_filename = str(img_number).zfill(6) + ".jpg"
                out_img_filepath = os.path.join(out_img_dirpath, out_img_filename)
                cv2.imwrite(out_img_filepath, frames_resized[img_in_batch_i])

            # Update pbar
            pbar.update(args.batch_size)

            # Update pbar description - average inference duration
            avg_duration = sum(inference_durations) / len(inference_durations) / args.batch_size
            if args.device == "cpu":
                pbar.set_description(f"Avg inference CPU duration: {'%.3f' % avg_duration}s")
            else:
                pbar.set_description(f"Avg inference real duration: {'%.3f' % avg_duration}s")

        del pbar
        print("Images annotated to", out_img_dirpath)

    except KeyboardInterrupt:
        print("KeyboardInterrupt: Stopped annotating images")

    if len(inference_durations):
        avg_duration = sum(inference_durations) / len(inference_durations) / args.batch_size
        if args.device == "cpu":
            print("Average inference CPU duration (per sample):", avg_duration)
        else:
            print("Average inference real duration (per sample):", avg_duration)

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
    parser.add_argument("-b", "--batch-size",   type=int,   default=1,
                        help="Inference batch size. Model needs to be dynamic to allow for it! Default 1")
    parser.add_argument("-n", "--number",       type=int,   default=-1,
                        help="number of frames to annotate. Default -1 to annotate all")
    parser.add_argument("-t", "--threshold",    type=float, default=DEFAULT_THRESHOLD,
                        help=f"score threshold. Default {DEFAULT_THRESHOLD}")
    parser.add_argument("-c", "--clean",        action="store_true",
                        help="remove the images dir after finish")
    parser.add_argument("-m", "--multi-thread", action="store_true",
                        help="multi-threaded inference (when on cpu)")
    args = parser.parse_args()

    main(args)
