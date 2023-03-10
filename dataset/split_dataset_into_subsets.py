"""
Splits a single (masked) dataset (with name provided as an argument) into
train/val/test subsets according to split requirements in common.datasets.
Should be only called if it was not yet split (eg. for DETRAC - detrac.py
processes the original data into COCO format, which I then reannotated, so
detrac.py should not be called again. Instead, only apply_masks.py and this file
should be executed after reannotated dataset was download from labelbox)
"""
import os
import json
import sys
import common
import random
from tqdm import tqdm


def split_dataset(dataset_name, data):
    print("Splitting into train, val and test...")
    random.seed(42)

    # Randomly reorder images if desired
    if "split_randomly" in common.datasets[dataset_name] and common.datasets[dataset_name]["split_randomly"] == True:
        random.shuffle(data["combined"]["images"])

    # Output - datasets split into train/val/test
    data = {
        "combined": data["combined"],
        "train": {"images": [], "annotations": [], "categories": []},
        "val": {"images": [], "annotations": [], "categories": []},
        "test": {"images": [], "annotations": [], "categories": []}
    }

    # Split data into train/val/test
    images_combined_backup = data["combined"]["images"].copy()

    # Split based on tags (simply remove from "combined" and append to
    # subset
    for img in tqdm(data["combined"]["images"].copy()):
        for subset, tags in common.datasets[img["dataset_name"]]["data_split_tags"].items():
            for tag in tags:
                if tag in img["file_name"]:
                    data[subset]["images"].append(img)
                    data["combined"]["images"].remove(img)

    # Split based on percentages
    distributed = 0 # Amount of images (from index 0) already distributed into subsets
    for subset in ["train", "val", "test"]:
        amount = int(len(data["combined"]["images"]) * common.datasets[dataset_name]["data_split"].get(subset, 0))
        data[subset]["images"] += data["combined"]["images"][distributed:distributed+amount]
        distributed += amount

    # Restore "combined"
    data["combined"]["images"] = images_combined_backup

    # Add annotations and categories to splits
    for subset in ["train", "val", "test"]:

        # Save image IDs per subset to img_ids
        img_ids = set()
        for img in data[subset]["images"]:
            img_ids.add(img["id"])

        # For each image, find all its annotations and add them to the
        # same subset
        for anno in data["combined"]["annotations"].copy():
            if anno["image_id"] in img_ids:
                data[subset]["annotations"].append(anno)

        data[subset]["categories"] = data["combined"]["categories"]

    return data


if __name__ == "__main__":
    assert len(sys.argv) == 2, "Please provide the dataset name as an argument"
    assert sys.argv[1] in common.datasets, f"Dataset {sys.argv[1]} not found"

    dataset_name = sys.argv[1]

    combined_dataset_filepath = os.path.join(
        common.paths.datasets_dirpath,
        common.datasets[dataset_name]["path"],
        common.gt_filenames["combined"])

    with open(combined_dataset_filepath) as f:
        data = {
            "combined": json.load(f)
        }

    data = split_dataset(dataset_name, data)

    # Save it to files
    print("Saving...")
    dataset_path = os.path.join(
        common.paths.datasets_dirpath, common.datasets[dataset_name]["path"])
    for key in ["train", "val", "test"]:
        gt_filepath = os.path.join(dataset_path, common.gt_filenames[key])
        with open(gt_filepath, 'w') as f:
            json.dump(data[key], f, indent=2)

        print(f"{key}: \t{len(data[key]['images'])} images \t{len(data[key]['annotations'])} annotations")