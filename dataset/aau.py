"""
Processes the AAU RainSnow dataset, which is in COCO format, mapping
classes and updating file paths. Saves ground truth in a COCO format
"""
import os
import json
import shutil
from tqdm import tqdm

# The script should be importable but also executable from the terminal...
if __name__ == '__main__':
    import common
else:
    from . import common


# Classes persent in the dataset: 1, 2, 3, 4, 6, 8, or in text format:
# person, bicycle, car, motorbike, bus, truck
aau_classes_map = {
    1: -1,                               # person
    2: common.classes_ids["bicycle"],    # bicycle
    4: common.classes_ids["motorcycle"], # motorbike
    3: common.classes_ids["car"],        # passenger car
    6: common.classes_ids["bus"],        # bus
    8: common.classes_ids["truck"]       # truck
}


def process_aau():
    # Initialize paths
    dataset_abs_dirpath = os.path.join(common.paths.datasets_dirpath, common.datasets["aau"]["path"])
    gt_json_abs_filepath = os.path.join(dataset_abs_dirpath, "aauRainSnow-rgb.json")

    # For some reason, Egensevej-5 mask has the wrong filename - fix that:
    if not os.path.exists(os.path.join(dataset_abs_dirpath, "Egensevej", "Egensevej-5-mask.png")):
        dirpath = os.path.join(dataset_abs_dirpath, "Egensevej")
        src = os.path.join(dirpath, "Egensevej-5.png")
        dst = os.path.join(dirpath, "Egensevej-5-mask.png")
        shutil.copy(src, dst)

    print(f"Loading data from {gt_json_abs_filepath}")
    with open(os.path.join(gt_json_abs_filepath)) as f:
        data = json.loads(f.read())

    print("Removing ignored images")
    for img in tqdm(data["images"].copy()):
        for ignore_str in common.datasets["aau"]["ignored_folders"]:
            if ignore_str in img["file_name"]:
                data["images"].remove(img)
                break

    print(f"Saving mask filepaths to ground truth data")
    for img in tqdm(data["images"]):

        sequence_dirpath = os.path.dirname(img["file_name"]) # Egensevej/Egensevej-1
        location_dirpath = os.path.dirname(sequence_dirpath) # Egensevej

        # Mask filename = Egensevej-1-mask.png
        mask_filename = os.path.basename(sequence_dirpath) + "-mask.png"

        # Mask filepath = Egensevej/Egensevej-1-mask.png
        img["mask"] = os.path.join(location_dirpath, mask_filename)

    print("Removing ignored annotations")
    img_ids = set()
    for img in data["images"]:
        img_ids.add(img["id"])
    for anno in tqdm(data["annotations"].copy()):
        if anno["image_id"] not in img_ids:
            data["annotations"].remove(anno)

    print("Removing invalid annotations")
    for anno in tqdm(data["annotations"].copy()):
        if anno["bbox"][2] < 0 or anno["bbox"][3] < 0:
            data["annotations"].remove(anno)

    print("Mapping classes")
    for anno in tqdm(data["annotations"].copy()):
        if aau_classes_map[anno["category_id"]] == -1:
            data["annotations"].remove(anno)
        else:
            anno["category_id"] = aau_classes_map[anno["category_id"]]

    # Convert data_dict to a list and all lists (bboxes and labels) to numpy arrays
    common.save_processed("aau", data)


if __name__ == "__main__":
    process_aau()
