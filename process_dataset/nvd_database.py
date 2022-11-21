"""
Put this script inside the NVD_Database folder so it sees GT/ and Images/
folders

Images should contain folders with names of cities and they all should contain
train/ and test/ folders (those should then contain seqNN folders)

WARNING: If there's no vehicle in a frame, let's say "000001234.jpg", label file
"000001234.txt" will not be created!
"""

import os
import shutil
from bs4 import BeautifulSoup
from datetime import datetime

GT_DIR = "GT"
IMAGES_DIR = "Images"
DST_DIR = "nvd_database"

CAR_CLASS = 3

# Original is 50x80 (width x height), but that seems to be too big too often...
# TODO make this adaptive - different for every city and based on X and Y
# coords
BBOX_WIDTH_PX = 30
BBOX_HEIGHT_PX = 40

FRAME_SIZES = {}

DATA = {}

"""
Src format:
Images:
    city:
        target:
            seq_nr:
                0001.jpg
GT:
    city:
        target:
            seq_nr:
                0001.jpg


YOLO format:

data:
    city:
        target:
            0001.jpg
            0001.txt
                (class_id, x_centre,  y_centre,  width,  height)

DATA variable format:
DATA = {
    city: {
        target: {
            frame_nr: [
                [class_id, x_centre, y_centre, w, h],
                [class_id, x_centre, y_centre, w, h]
                ...
}
"""


def timestamp():
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


def out(*args):
    print(f"{timestamp()}: ", end="")
    print(*args)


def getSequenceStartingFrameNr(city, target, seq_nr):
    path = os.path.join(IMAGES_DIR, city, target, "seq" + str(seq_nr))
    files = os.listdir(path)

    # Not sure if the result is alphabetically sorted, so we rather do it
    # manually
    files.sort()
    return int(files[0].split(".")[0])


def parseXml(city, target, file_path, seq_nr):
    with open(file_path) as f:
        soup = BeautifulSoup(f.read(), "lxml")

        if not city in FRAME_SIZES.keys():
            FRAME_SIZES[city] = {}
            for metadata in soup.find_all("metadata"):
                if "width" in metadata.attrs:
                    FRAME_SIZES[city]["w"] = int(metadata["width"])
                if "height" in metadata.attrs:
                    FRAME_SIZES[city]["h"] = int(metadata["height"])


        # [<data frame="1" x="449" y="119" />, ...]
        objects = soup.find_all("data")

        for obj in objects:
            # The XML file contains relative frame numbers - for the
            # sequence (starting at 1)
            rel_frame_nr = int(obj["frame"]) - 1

            # But we need absolute frame number
            seq_starting_frame_nr = getSequenceStartingFrameNr(city, target, seq_nr)
            abs_frame_nr = seq_starting_frame_nr + rel_frame_nr
            abs_frame_nr = str(abs_frame_nr).zfill(9)

            x = int(obj["x"]) / FRAME_SIZES[city]["w"]
            y = int(obj["y"]) / FRAME_SIZES[city]["h"]

            bbox_w = BBOX_WIDTH_PX / FRAME_SIZES[city]["w"]
            bbox_h = BBOX_HEIGHT_PX / FRAME_SIZES[city]["h"]

            # TODO Multiplier for width and height based on X and (or) Y coords?
            annotation = [CAR_CLASS, x, y, bbox_w , bbox_h]

            if not abs_frame_nr in DATA[city][target].keys():
                DATA[city][target][abs_frame_nr] = []

            DATA[city][target][abs_frame_nr].append(annotation)


if __name__ == "__main__":
    if not os.path.exists(IMAGES_DIR) or not os.path.exists(GT_DIR):
        print(f"{IMAGES_DIR}/ or {GT_DIR}/ folder missing!")
        exit(0)

    # Get the data
    out("Getting data")
    for city in os.listdir(GT_DIR):
        city = os.path.basename(city)
        city_path = os.path.join(GT_DIR, city)
        out(f"Getting data for city {city}")

        DATA[city] = {}

        for target in os.listdir(city_path):
            target = os.path.basename(target)
            target_path = os.path.join(city_path, target)
            out(f"Getting data for city {city}, target {target}")

            DATA[city][target] = {}

            for f in os.listdir(target_path):
                f = os.path.basename(f)
                file_path = os.path.join(target_path, f)
                seq_nr = int(f.split("_")[-2])
                out(f"Getting data for city {city}, target {target}, sequence {seq_nr}")

                parseXml(city, target, file_path, seq_nr)

    out("Saving data")

    # Save the data
    if not os.path.exists(DST_DIR):
        os.mkdir(DST_DIR)

    # ["California", ...]
    for city in list(DATA.keys()):
        out(f"Saving data for city {city}")
        city_path = os.path.join(DST_DIR, city)
        if not os.path.exists(city_path):
            os.mkdir(city_path)

        # ["test", "train"]
        for target in list(DATA[city].keys()):
            out(f"Saving data for city {city}, target {target}")
            target_path = os.path.join(city_path, target)
            if not os.path.exists(target_path):
                os.mkdir(target_path)

            # ["000000001", ...]
            for frame_nr in list(DATA[city][target].keys()):
                filepath = os.path.join(target_path, frame_nr + ".txt")

                with open(filepath, "w") as f:

                    # [ [cls, x, y, w, h], ...
                    for [cls, x, y, w, h] in DATA[city][target][frame_nr]:
                        f.write(f"{cls} {x} {y} {w} {h}\n")

    out("Copying images")
    for city in os.listdir(IMAGES_DIR):
        city = os.path.basename(city)
        city_path = os.path.join(IMAGES_DIR, city)
        out(f"Copying images for city {city}")

        for target in os.listdir(city_path):
            target = os.path.basename(target)
            target_path = os.path.join(city_path, target)
            out(f"Copying images for city {city}, target {target}")

            dst_path = os.path.join(DST_DIR, city, target)

            # Note: I understand this is much slower than copying in shell, but
            # this way it should work with any operating system
            for seq in os.listdir(target_path):
                seq = os.path.basename(seq)
                seq_path = os.path.join(target_path, seq)
                out(f"Copying images for city {city}, target {target}, sequence {seq}")

                for f in os.listdir(seq_path):
                    f = os.path.basename(f)
                    src_filepath = os.path.join(seq_path, f)
                    dst_filepath = os.path.join(dst_path, f)
                    shutil.copy(src_filepath, dst_filepath)

    out("All done")
