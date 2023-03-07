"""
Reads all annotation of a dataset (name provided as an argument) and draws them
on images and saves the images with bounding boxes to a separate folder
"""
import sys
import json
import os
import shutil
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

# The script should be importable but also executable from the terminal...
if __name__ == '__main__':
    import common
else:
    from . import common


def visualize_dataset(dataset_name):

    print(f"Visualizing dataset {dataset_name}")

    if dataset_name not in common.datasets.keys():
        print(f"Dataset with name {dataset_name} not found in common.datasets")
        return
    dst_dirpath = os.path.join(common.paths.datasets_dirpath, "visualized_" + dataset_name)
    if os.path.exists(dst_dirpath):
        shutil.rmtree(dst_dirpath)
    os.mkdir(dst_dirpath)

    dataset_abs_dirpath = os.path.join(common.paths.datasets_dirpath, common.datasets[dataset_name]["path"])

    dataset_gt_filepath = os.path.join(dataset_abs_dirpath, common.gt_filenames["combined"])

    with open(dataset_gt_filepath, "r") as f:
        data = json.loads(f.read())

    annos = {} # Key = img ID, val = list of annotations
    for anno in data["annotations"]:
        img_id = anno["image_id"]
        if img_id not in annos:
            annos[img_id] = [anno]
        else:
            annos[img_id].append(anno)

    font = ImageFont.load_default()

    for img in tqdm(data["images"]):

        img_id = img["id"]

        old_abs_filepath = os.path.join(dataset_abs_dirpath, img["file_name"])

        rel_filepath = os.path.dirname(img["file_name"]) # Relative to dataset
        new_filename = str(img_id).zfill(6) + "_" + os.path.basename(old_abs_filepath)
        new_abs_dirpath = os.path.join(dst_dirpath, rel_filepath)
        new_abs_filepath = os.path.join(new_abs_dirpath, new_filename)

        if not os.path.exists(new_abs_dirpath):
            os.makedirs(new_abs_dirpath)

        assert os.path.exists(old_abs_filepath), f"{old_abs_filepath} file does not exist"
        img = Image.open(old_abs_filepath)
        draw = ImageDraw.Draw(img)
        for anno in annos.get(img_id, []):
            bbox = anno["bbox"]
            bbox[2] += bbox[0] # width -> x2
            bbox[3] += bbox[1] # height -> y2
            cls = anno["category_id"]
            if cls == common.classes_ids["car"]:
                color = (0, 255, 0)
            elif cls == common.classes_ids["mask"]:
                color = (0, 0, 0)
            else:
                color = (0, 0, 255)
            text = f"{common.classes_names[cls]} (bbox {anno['id']})"

            draw.rectangle(bbox, outline=color)
            # TODO set bigger font size
            draw.text((bbox[0], bbox[3]), text, font=font, fill=(255, 0, 0))

        img.save(new_abs_filepath)

        # Previously done by opencv, but PIL is faster (3x)
        # img = cv2.imread(old_abs_filepath)
        # img = cv2.rectangle(img, bbox[:2], bbox[2:], color, 1)
        # img = cv2.putText(img, text, (bbox[0], bbox[3]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
        # cv2.imwrite(new_abs_filepath, img)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Please provide a dataset name as an argument")
        exit(1)

    # import cProfile
    # import pstats
    # pr = cProfile.Profile()
    # pr.enable()
    # visualize_dataset(sys.argv[1])
    # pr.disable()
    # ps = pstats.Stats(pr)
    # ps.sort_stats('cumtime')
    # ps.print_stats(10)

    visualize_dataset(sys.argv[1])
