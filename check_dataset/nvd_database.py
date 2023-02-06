import os
import shutil
import cv2
from datetime import datetime


DATA_DIR = "nvd_database"

FPS = 30

TMP_DIR = "tmp"
DST_DIR = "check"


def timestamp():
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


def out(*args):
    print(f"{timestamp()}: ", end="")
    print(*args)


if __name__ == "__main__":

    if not os.path.exists(DST_DIR):
        os.mkdir(DST_DIR)

    for city in os.listdir(DATA_DIR):
        city = os.path.basename(city)
        city_path = os.path.join(DATA_DIR, city)
        out(f"Annotating images for city {city}")

        # Ensure TMP_DIR is empty
        if os.path.exists(TMP_DIR):
            shutil.rmtree(TMP_DIR)
        os.mkdir(TMP_DIR)

        for target in os.listdir(city_path):
            target = os.path.basename(target)
            target_path = os.path.join(city_path, target)

            # Get first and last frame name (as integers)
            names = os.listdir(target_path)
            names.sort()
            first = int(os.path.basename(names[0]).split(".")[0])
            last = int(os.path.basename(names[-1]).split(".")[0])

            for name in range(first, last + 1):

                name_str = str(name).zfill(9)

                img_filepath = os.path.join(target_path, name_str + ".jpg")

                # Frames can be in different targets - ignore it if a frame is
                # not present
                if not os.path.exists(img_filepath):
                    continue

                img = cv2.imread(img_filepath)
                img_h, img_w, _ = img.shape

                label_filepath = os.path.join(target_path, name_str + ".txt")

                # Ignore this part if there's no label for the photo
                if os.path.exists(label_filepath):
                    with open(label_filepath) as f:
                        for l in f.readlines():

                            # This is in YOLO format:
                            [_, x, y, w, h] = [float(val) for val in l.split(" ")]
                            w = img_w * w
                            h = img_h * h
                            center_x = img_w * x
                            center_y = img_h * y
                            x1 = int(center_x - w / 2)
                            x2 = int(center_x + w / 2)
                            y1 = int(center_y - h / 2)
                            y2 = int(center_y + h / 2)

                            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 1)

                new_filepath = os.path.join(TMP_DIR, name_str + ".jpg")
                cv2.imwrite(new_filepath, img)

        out(f"Converting images to video for city {city}")

        # TODO Might only work on linux:
        cmd = f"ffmpeg -loglevel warning -stats -framerate {FPS} -pattern_type glob -i '{TMP_DIR}/*.jpg' -c:v libx264 -pix_fmt yuv420p {DST_DIR}/{city}.mp4"
        os.system(cmd)

        out(f"Done: city {city}")

    if os.path.exists(TMP_DIR):
        shutil.rmtree(TMP_DIR)

    out("All done")

