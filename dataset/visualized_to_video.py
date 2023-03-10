"""
Converts visualized dataset to video (or videos). It browses the directory with
visualized images and for each folder containing images (directly), it converts
them to a video. All child directories are browsed recursively.
"""
import os
import cv2
import shutil
import sys
import time
from tqdm import tqdm
from threading import Thread

import common


FPS = 25
PBAR = None
TOTAL_DIRS_DISCOVERED = 0

THREADS_LIST = []
THREADS_RUNNING = 0

def dir_to_videos(src_abs_dirpath, dst_abs_dirpath):
    global TOTAL_DIRS_DISCOVERED
    global THREADS_LIST
    global THREADS_RUNNING

    """Recursively converts all images in `src_abs_dirpath` directory to videos.
    Each folder in `src_abs_dirpath` tree containing any number of images will
    result in one video. As said, this function is called for each subdirectory.

    Args:
        src_abs_dirpath (str): Source dir
        dst_abs_dirpath (str): Destination dir
    """

    # Create output dir
    if os.path.exists(dst_abs_dirpath):
        shutil.rmtree(dst_abs_dirpath)
    os.mkdir(dst_abs_dirpath)

    files = os.listdir(src_abs_dirpath)

    # First, process all directories recursively
    for f in files:
        src_abs_path = os.path.join(src_abs_dirpath, f)
        if os.path.isdir(src_abs_path):
            dst_abs_path = os.path.join(dst_abs_dirpath, f)

            # Create a thread
            t = Thread(target=dir_to_videos, args=(src_abs_path, dst_abs_path))
            t.start()
            THREADS_LIST.append(t)

    # Then, keep only ".jpg" and ".png" files in this directory (in the
    # variable. Of course we don't delete them from the disk)
    for f in files.copy():
        if not f.endswith(".jpg") and not f.endswith(".png"):
            files.remove(f)

    files.sort()
    
    if len(files):

        TOTAL_DIRS_DISCOVERED += 1
        PBAR.__setattr__("total", TOTAL_DIRS_DISCOVERED)

        # Wait if there are too many threads
        while THREADS_RUNNING >= common.max_threads:
            time.sleep(0.1)
            
        THREADS_RUNNING += 1

        src_abs_filepaths = [os.path.join(src_abs_dirpath, f) for f in files]

        # Put the video one directory up, to be named as the directory in which
        # the images were
        dst_abs_filepath = dst_abs_dirpath + ".mp4"

        h, w, _ = cv2.imread(src_abs_filepaths[0]).shape

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        video = cv2.VideoWriter(dst_abs_filepath, fourcc, FPS, (w, h))

        for src_abs_filepath in src_abs_filepaths:
            video.write(cv2.imread(src_abs_filepath))

        video.release()

        PBAR.update(1)
    
        THREADS_RUNNING -= 1


def remove_empty_dirs(root_dirpath):
    """Recursively remove all empty directories in a directory.

    Args:
        root_dirpath (str): Absolute path to a root folder from where to begin
    """
    for f in os.listdir(root_dirpath):
        abs_path = os.path.join(root_dirpath, f)
        if os.path.isdir(abs_path):
            if os.listdir(abs_path) == []:
                os.rmdir(abs_path)
            else:
                remove_empty_dirs(abs_path)


def visualized_to_video(dataset_name):

    """Converts visualized dataset (masked and with annotations) to a video. If
    the visualized images are separated into directories, each directory will
    make an individual video.

    Args:
        dataset_name (str): Dataset name
    """
    global PBAR
    global THREADS_LIST

    print(f"Converting visualized dataset {dataset_name} to video")

    PBAR = tqdm(desc="Processing directories")

    if dataset_name not in common.datasets.keys():
        print(f"Dataset with name {dataset_name} not found in common.datasets")
        return

    src_dirpath = os.path.join(common.paths.datasets_dirpath, "visualized_" + dataset_name)
    dst_dirpath = os.path.join(common.paths.datasets_dirpath, "visualized_" + dataset_name + "_videos")

    if os.path.exists(dst_dirpath):
        shutil.rmtree(dst_dirpath)

    assert os.path.exists(src_dirpath), "First, run `visualize_dataset.py`"
    files = os.listdir(src_dirpath)
    assert len(files) != 0, "First, run `visualize_dataset.py detrac`"

    dir_to_videos(src_dirpath, dst_dirpath)

    while len(THREADS_LIST):
        THREADS_LIST[0].join()
        THREADS_LIST.remove(THREADS_LIST[0])

    remove_empty_dirs(dst_dirpath)

    del PBAR

    tqdm.write("All done")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Please provide a dataset name as an argument")
        exit(1)

    # import cProfile
    # import pstats
    # pr = cProfile.Profile()
    # pr.enable()
    # visualized_to_video(sys.argv[1])
    # pr.disable()
    # ps = pstats.Stats(pr)
    # ps.sort_stats('cumtime')
    # ps.print_stats(10)

    visualized_to_video(sys.argv[1])
