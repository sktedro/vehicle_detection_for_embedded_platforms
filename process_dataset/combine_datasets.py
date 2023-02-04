"""
Combine datasets to a single dataset in a pickle file

Output format example:
```py
{
    "train": [
        {
            # Note: no ID needed as the index can serve as an ID
            dataset_name: "dataset_name",
            filename: "dataset_name/images/0001.jpg",
            width: 1280,
            height: 720,
            ann: {
                bboxes: ndarray([x1, y1, x2, y2], ...),
                labels: array(1, ...)
            }
            },
           ...
    ],
    "val": [...],
    "test": [...],
    "datasets": [
        {
            "name": "dataset_name",
            "rel_path": "dataset_name"
            },
        ..
    ]
```
"""

import os
import pickle
import random
import numpy as np

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
    # "train": 0.6,
    # "val": 0.2,
    # "test": 0.2
    "train": 0.9,
    "val": 0.1,
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
    dataset = {
        "datasets": [],
        "train": [],
        "val": [],
        "test": []
    }

    for name in list(common.datasets.keys()):
        print(f"Reading {name}")
        pickle_filepath = os.path.join(common.datasets_path, common.datasets[name]["path"], "gt.pickle")
        
        # Open the dataset's pickle file and load and process data
        with open(pickle_filepath, "rb") as f:
            data = pickle.load(f)

            # Randomly reorder images if desired
            if random_data_distribution:
                random.shuffle(data)
            
            # TODO TODO remove
            # if name == "mio-tcd":
            #     print("Removing 3/4 of MIO-TCD dataset")
            #     data = data[:len(data)//4]

            # Set dataset_name and update filename (relative filepath)
            # filename needs to be updated to be relative to the datasets folder
            for img in data:
                img["dataset_name"] = name
                img["filename"] = os.path.join(common.datasets[name]["path"], img["filename"])

            # Ignore all images that have no annotation! Because mmdetection
            # can't handle that. Maybe there's a way to fix that but right now
            # I'm not debugging it...
            for i in reversed(range(len(data))):
                if len(data[i]["ann"]["labels"]) == 0:
                    del data[i]

            # Split data into train/val/test
            data_len = len(data)
            train_val_split_index = int(data_len * data_distribution["train"])
            val_test_split_index = int(data_len * (data_distribution["train"] + data_distribution["val"]))
            dataset["train"] += data[:train_val_split_index]
            dataset["val"] +=   data[train_val_split_index:val_test_split_index]
            dataset["test"] +=  data[val_test_split_index:]

            dataset["datasets"].append({
                "name": name,
                "rel_dataset_path": common.datasets[name]["path"]
            })

    print(f"Read {len(dataset['train'])} training, {len(dataset['val'])} validation and {len(dataset['test'])} testing images")

    with open(common.dataset_pickle_filepath, 'wb') as f:
        pickle.dump(dataset, f, protocol=common.pickle_file_protocol)

    with open(common.train_pickle_filepath, 'wb') as f:
        pickle.dump(dataset["train"], f, protocol=common.pickle_file_protocol)
        
    with open(common.val_pickle_filepath, 'wb') as f:
        pickle.dump(dataset["val"], f, protocol=common.pickle_file_protocol)

    with open(common.test_pickle_filepath, 'wb') as f:
        pickle.dump(dataset["test"], f, protocol=common.pickle_file_protocol)

    print("All saved and done")


if __name__ == "__main__":
    combineDatasets()