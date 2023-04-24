"""
Splits the test subset into several by dataset names. Eg. if the test subset
contains images from DETRAC and MIO-TCD datasets, two files will be created -
test_detrac.json and test_mio-tcd.json
"""

import copy
import json
import os

# The script should be importable but also executable from the terminal...
if __name__ == '__main__':
    import common
else:
    try:
        from . import common
    except:
        import sys
        sys.path.append(os.path.dirname(__file__))
        import common


def split_test_by_dataset():

    print("Reading data")
    test_filename = common.gt_combined_filenames["test"]
    test_filepath = os.path.join(common.paths.datasets_dirpath, test_filename)
    with open(test_filepath) as f:
        data = json.load(f)

    print("Extracting image IDs")
    # Image ids as a set in dict by keys being dataset names
    image_ids = {}
    for img in data["images"]:
        if img["dataset_name"] not in image_ids:
            image_ids[img["dataset_name"]] = set()
        image_ids[img["dataset_name"]].add(img["id"])

    dataset_names = list(image_ids.keys())
    print("Datasets:", dataset_names)

    # `data` copy for each dataset name, but without images and annotations (those will be added later)
    data_split = {}
    for dataset_name in dataset_names:
        if dataset_name not in data_split:
            data_split[dataset_name] = {}
            for key in data:
                if key not in ["images", "annotations"]:
                    data_split[dataset_name][key] = copy.deepcopy(data[key])
            data_split[dataset_name]["images"] = []
            data_split[dataset_name]["annotations"] = []

    # Distribute images and annotations from `data` to `data_split`
    print("Splitting images")
    for img in data["images"]:
        data_split[img["dataset_name"]]["images"].append(img)
    print("Splitting annotations")
    for anno in data["annotations"]:
        for dataset_name in image_ids:
            if anno["image_id"] in image_ids[dataset_name]:
                data_split[dataset_name]["annotations"].append(anno)
                break

    # Finally, save the files
    print("Saving files")
    for dataset_name in dataset_names:
        new_filename = ("".join(test_filename.split(".")[:-1])
                        + "_" + dataset_name + "."
                        + test_filename.split(".")[-1])
        new_filepath = os.path.join(common.paths.datasets_dirpath, new_filename)
        with open(new_filepath, "w") as f:
            json.dump(data_split[dataset_name], f, indent=2)

    print("All done")


if __name__ == "__main__":
    split_test_by_dataset()