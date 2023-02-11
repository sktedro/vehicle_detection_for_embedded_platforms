"""
Combine datasets to a single dataset in COCO format
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


####################
##### Settings #####
####################


# Provide data split values for train, val, and testing data as percentage / 100
data_distribution = {
    "train": 0.95,
    "val": 0.05,
    "test": 0
}
# Make sure it sums up to 1...
assert sum(data_distribution[key] for key in data_distribution.keys()) == 1

# Select if you want your splits to be continuous vs random
# Eg. of continuous: train will contain 0.jpg, 1.jpg, 2.jpg, ...
random_data_distribution = True

# Set random seed
random.seed(42)


################################
##### Combine the datasets #####
################################

def combineDatasets():

    # Output - datasets split into train/val/test
    data_split = {
        "train": {"images": [], "annotations": [], "categories": []},
        "val": {"images": [], "annotations": [], "categories": []},
        "test": {"images": [], "annotations": [], "categories": []}
    }

    img_id_counter = 0
    anno_id_counter = 0

    # For each dataset, split it into train/val/test
    print("Reading datasets")
    for dataset_name in tqdm(list(common.datasets.keys())):

        gt_filepath = os.path.join(common.datasets_dirpath, common.datasets[dataset_name]["path"], common.gt_filename)
        
        img_id_map = {} # Mapping old image IDs to new ones

        # Open the dataset's pickle file and load and process data
        with open(gt_filepath) as f:
            data = json.loads(f.read())

            # Randomly reorder images if desired
            if random_data_distribution:
                random.shuffle(data["images"])

            # Update images
            for img in data["images"]:
                old_img_id = img["id"]
                new_img_id = img_id_counter
                img_id_map[old_img_id] = new_img_id

                img["id"] = new_img_id
                img["dataset_name"] = dataset_name

                img_id_counter += 1

            # Update annotations' IDs and img IDs
            for anno in data["annotations"]:
                anno["image_id"] = img_id_map[anno["image_id"]]
                anno["id"] = anno_id_counter

                anno_id_counter += 1

            # Split data into train/val/test
            train_val_split_index = int(len(data["images"]) * data_distribution["train"])
            val_test_split_index = int(len(data["images"]) * (data_distribution["train"] + data_distribution["val"]))

            data_split["train"]["images"] += data["images"][:train_val_split_index]
            data_split["val"]["images"]   += data["images"][train_val_split_index:val_test_split_index]
            data_split["test"]["images"]  += data["images"][val_test_split_index:]

            # Add annotations and update their img IDs
            for subset in ["train", "val", "test"]:

                # Save image IDs per subset to img_ids_split
                img_ids = set()
                for img in data_split[subset]["images"]:
                    img_ids.add(img["id"])

                # For each image, find all its annotations and add them to the
                # same subset
                for anno in data["annotations"]:
                    if anno["image_id"] in img_ids:
                        data_split[subset]["annotations"].append(anno)

                data_split[subset]["categories"] = data["categories"]

    # Only keep "images", "annotations" and "categories" in the datasets
    for subset in ["train", "val", "test"]:
        for key in list(data_split[subset].keys()):
            if key not in ["images", "annotations", "categories"]:
                del data_split[subset][key]

    # Combine train/val/test to a combined dataset
    data_combined = {}
    for key in ["images", "annotations", "categories"]:
        data_combined[key] = data_split["train"][key] + data_split["val"][key] + data_split["test"][key]

    for subset in ["train", "val", "test"]:
        print(f"{subset}: \t{len(data_split[subset]['images'])} images \t{len(data_split[subset]['annotations'])} annotations")
    print(f"Total: \t{len(data_combined['images'])} images \t{len(data_combined['annotations'])} annotations")

    with open(common.dataset_filepath, "w") as f:
        f.write(json.dumps(data_combined))
        
    with open(common.dataset_train_filepath, "w") as f:
        f.write(json.dumps(data_split["train"]))

    with open(common.dataset_val_filepath, "w") as f:
        f.write(json.dumps(data_split["val"]))

    with open(common.dataset_test_filepath, "w") as f:
        f.write(json.dumps(data_split["test"]))

    print("All saved and done")

if __name__ == "__main__":
    combineDatasets()