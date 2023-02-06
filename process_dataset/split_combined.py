"""
Splits combined dataset.pickle to files by origin dataset
Eg. if dataset.pickle contains annotations for MIO-TCD and MTID datasets,
files will be created in folder common.split_datasets_path: MIO-TCD_train.pickle, MIO-TCD_test.pickle, ...

dataset.pickle format:
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

Example output format:
[
    {
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
"""

import pickle
import os
import shutil

# The script should be importable but also executable from the terminal...
if __name__ == '__main__':
    import common
else:
    from . import common as common


################################
##### Combine the datasets #####
################################


def splitCombined():

    print("Splitting dataset.pickle into files by subset and origin dataset")

    assert os.path.exists(common.dataset_pickle_filepath)

    output = {} # = { "dataset_name": { "train": [], ...}, "dataset_name_2": ...}

    # Initialize output var by dataset names in common
    for dataset_name in list(common.datasets.keys()):
        output[dataset_name] = {
            "train": [],
            "test": [],
            "val": []
        }

    # Load the annotations into the output var
    with open(common.dataset_pickle_filepath, "rb") as datasets_f:
        datasets = pickle.load(datasets_f)
        for subset in ["train", "test", "val"]:
            for annotation in datasets[subset]:
                output[annotation["dataset_name"]][subset].append(annotation)


    # Prepare the output directory
    if os.path.exists(common.split_datasets_path):
        shutil.rmtree(common.split_datasets_path)
    os.mkdir(common.split_datasets_path)

    # Save the data
    for dataset_name in list(output.keys()):
        for subset in list(output[dataset_name].keys()):
            filename = f"{dataset_name}_{subset}.pickle"
            filepath = os.path.join(common.split_datasets_path, filename)
            with open(filepath, 'wb') as output_f:
                pickle.dump(output[dataset_name][subset], output_f, protocol=common.pickle_file_protocol)
    
    print("All done")

if __name__ == "__main__":
    splitCombined()