import fiftyone as fo
import common
from tqdm import tqdm

"""
Runs fiftyone. If a dataset was not yet created (or if param delete=False), it
loads a dataset (reads paths from common.py), adds tags to the samples (dataset
name) and then runs fiftyone.
"""

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
    dataset = fo.Dataset.from_dir(
        name=dataset_name,
        data_path=common.paths.datasets_dirpath,
        labels_path=common.dataset_filepath,
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
