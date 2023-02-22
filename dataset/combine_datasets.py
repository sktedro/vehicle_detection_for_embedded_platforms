import os
import json
import random
from tqdm import tqdm

# The script should be importable but also executable from the terminal...
if __name__ == '__main__':
    import common
else:
    from . import common as common


"""
Combine all subsets of all datasets to 4 json files: combined, train, val and
test
"""
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

    print("Reading data")
    for dataset_name in tqdm(list(common.datasets.keys())):
        for subset in ["train", "val", "test"]:
            gt_filepath = os.path.join(
                    common.paths.datasets_dirpath, 
                    common.datasets[dataset_name]["path"], 
                    common.gt_filenames[subset])
            with open(gt_filepath) as f:
                data_split[subset][dataset_name] = json.loads(f.read())

    img_id_counter = 0
    anno_id_counter = 0
    img_id_map = {} # Mapping old image IDs to new ones

    input_subsets = ["train", "val", "test"]
    input_dataset_names = list(data_split[subset].keys())

    print("Combining...")
    progress_bar = tqdm(total=len(input_subsets) * len(input_dataset_names))
    for subset in input_subsets:
        for dataset_name in input_dataset_names:
            progress_bar.update(1)
            progress_bar.refresh()

            # Update image ID for each img and save it
            for img in data_split[subset][dataset_name]["images"].copy():

                # Update image ID and save it to the img ID map
                img_id_map[img["id"]] = img_id_counter
                img["id"] = img_id_counter

                # Save the image
                data_combined[subset]["images"].append(img)
                data_combined["combined"]["images"].append(img)

                data_split[subset][dataset_name]["images"].remove(img) # Delete processed to optimize
                img_id_counter += 1

            # Get all annotations in this image, update image ID and
            # annotation ID and save it
            for anno in data_split[subset][dataset_name]["annotations"].copy():

                # Update image ID and anno ID
                anno["image_id"] = img_id_map[anno["image_id"]]
                anno["id"] = anno_id_counter

                # Save the anno
                data_combined[subset]["annotations"].append(anno)
                data_combined["combined"]["annotations"].append(anno)

                data_split[subset][dataset_name]["annotations"].remove(anno) # Delete processed to optimize
                anno_id_counter += 1

            if data_combined[subset]["categories"] == []:
                data_combined[subset]["categories"] = data_split[subset][dataset_name]["categories"]

    del progress_bar

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
