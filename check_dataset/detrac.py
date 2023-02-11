"""
Put this file in the parent folder of the visualized detrac dataset (generated
by visualize_dataset.py)

Takes frames of visualized dataset and converts them to videos, one video per sequence
"""

import os
import cv2
import shutil
from tqdm import tqdm
import ffmpeg # ffmpeg-python package

dirname = "visualized_detrac"
output_dirname = "visualized_detrac_videos"
fps = 25

assert os.path.exists(dirname)
files = os.listdir(dirname)
assert len(files) != 0

# Create output dir
if os.path.exists(output_dirname):
    shutil.rmtree(output_dirname)
os.mkdir(output_dirname)

sequences = {} # key = sequence number, val = list of full filenames

for f in files:
    seq_number = f.split("_")[1]
    if seq_number not in sequences.keys():
        sequences[seq_number] = []
    sequences[seq_number].append(f)

print("Converting to video")

for seq_number in tqdm(list(sequences.keys())):
    sequences[seq_number].sort()
    video_filepath = os.path.join(output_dirname, seq_number + ".avi")
    compressed_video_filepath = os.path.join(output_dirname, seq_number + "_compressed.mp4")

    frame_tmp = cv2.imread(os.path.join(dirname, sequences[seq_number][0]))
    h, w, _ = frame_tmp.shape

    video = cv2.VideoWriter(video_filepath, 0, fps, (w, h))

    for img_filename in tqdm(sequences[seq_number], leave=False):
        video.write(cv2.imread(os.path.join(dirname, img_filename)))

    # cv2.destroyAllWindows() # Crashes if called

    video.release()

    # Compress the video a bit... (ultrafast h264 compresses to ~4% size)
    stream = ffmpeg.input(video_filepath)
    stream = ffmpeg.output(stream, compressed_video_filepath, vcodec="libx264", preset="ultrafast")
    stream = stream.global_args("-loglevel", "error")
    stream.run()

    # Remove the big video
    os.remove(video_filepath)