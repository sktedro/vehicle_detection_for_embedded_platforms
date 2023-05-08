"""
Combine all subsets of all datasets to 4 json files: combined, train, val and
test
"""
import os
import json
import random
from tqdm import tqdm

# The script should be importable but also executable from the terminal...
if __name__ == '__main__':
    import common
else:
    from . import common as common


def combineDatasets():
    # Output
    data_combined = {
        "combined": {"images": [], "annotations": [], "categories": []},
        "train": {"images": [], "annotations": [], "categories": []},
        "val": {"images": [], "annotations": [], "categories": []},
        "test": {"images": [], "annotations": [], "categories": []},
    }

    # All subsets of all datasets
    data_split = {"train": {}, "val": {}, "test": {}}

    # Load all gt files and check if categories are equal in all gt files
    categories = None
    progress_bar = tqdm(total=len(data_split.keys()) * len(common.datasets), desc="Loading data")
    for dataset_name in common.datasets:
        for subset in ["train", "val", "test"]:
            gt_filepath = os.path.join(
                    common.paths.datasets_dirpath,
                    common.datasets[dataset_name]["path"], 
                    common.gt_filenames[subset])
            with open(gt_filepath) as f:
                data_split[subset][dataset_name] = json.loads(f.read())
            if categories == None:
                categories = data_split[subset][dataset_name]["categories"]
            else:
                assert categories == data_split[subset][dataset_name]["categories"], f"Categories in datasets to combine don't match: \nReference={categories}, \n{dataset_name}({subset})={data_split[subset][dataset_name]['categories']}"
            progress_bar.update(1)

    del progress_bar

    img_id_map = {} # Mapping old image IDs to new ones
    img_id_counter = 0
    anno_id_counter = 0

    # Optimize by first creating a set of image IDs in each set and then going through
    # the annotations list only once?
    # Wait what? Why do I have to read the annotations?
    progress_bar = tqdm(total=len(data_split.keys()) * len(common.datasets), desc="Combining")
    for subset in data_split:
        for dataset_name in common.datasets:
            progress_bar.update(1)

            # Update image ID for each img and save it
            for img in data_split[subset][dataset_name]["images"]:

                # Update image ID and save it to the img ID map
                img_id_map[img["id"]] = img_id_counter
                img["id"] = img_id_counter
                # Update file_name to be relative to datasets path, not dataset path
                img["file_name"] = os.path.join(
                    common.datasets[img["dataset_name"]]["path"],
                    img["file_name"])

                img_id_counter += 1

            # Get all annotations in this image, update image ID and
            # annotation ID and save it
            for anno in data_split[subset][dataset_name]["annotations"]:

                # Update image ID and anno ID
                anno["image_id"] = img_id_map[anno["image_id"]]
                anno["id"] = anno_id_counter

                anno_id_counter += 1

            data_combined[subset]["images"] += data_split[subset][dataset_name]["images"]
            data_combined[subset]["annotations"] += data_split[subset][dataset_name]["annotations"]
            data_combined[subset]["categories"] = categories

    del progress_bar

    for subset in ["train", "val", "test"]:
        data_combined["combined"]["images"] += data_combined[subset]["images"]
        data_combined["combined"]["annotations"] += data_combined[subset]["annotations"]
    data_combined["combined"]["categories"] = categories

    # Print amount of images and annotations and save the file
    print("Saving...")
    for subset in ["train", "val", "test", "combined"]:
        print(f"{subset.ljust(8)} {len(data_combined[subset]['images'])} images \t{len(data_combined[subset]['annotations'])} annotations")

        filepath = os.path.join(
                common.paths.datasets_dirpath, 
                common.gt_combined_filenames[subset])
        with open(filepath, "w") as f:
            f.write(json.dumps(data_combined[subset], indent=2))

if __name__ == "__main__":
    combineDatasets()
