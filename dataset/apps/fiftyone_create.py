"""Reads all datasets and creates a fiftyone dataset while adding a tag to each
image: dataset name
"""
import fiftyone as fo
import os
import sys
from tqdm import tqdm

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import common


dataset_name = "dataset"


def fiftyone_create():

    if dataset_name in fo.list_datasets():
        s = input(f"Are you sure you want to overwrite the existing dataset '{dataset_name}'? (y/n): ")
        if s == "y":
            dataset = fo.load_dataset(dataset_name) 
            dataset.delete()
            print("Dataset deleted")
        else:
            print("Deletion aborted. Exiting")
            return

    print("Creating dataset")
    gt_filepath = os.path.join(common.paths.datasets_dirpath, common.gt_combined_filenames["combined"])
    dataset = fo.Dataset.from_dir(
        name=dataset_name,
        data_path=common.paths.datasets_dirpath,
        labels_path=gt_filepath,
        dataset_type=fo.types.COCODetectionDataset
    )

    print("Updating tags")
    for sample in tqdm(dataset):
        ds_name = sample["filepath"].split(common.paths.datasets_dirpath)[1].split("/")[0]
        sample.tags.append(ds_name)
        sample.save()

    if not dataset.persistent:
        dataset.persistent = True

    view = dataset.view()
    print(view)

    print("All done")


if __name__ == "__main__":
    fiftyone_create()
