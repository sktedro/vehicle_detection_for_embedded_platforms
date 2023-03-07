"""Starts fiftyone application
"""
import fiftyone as fo


dataset_name = "dataset"


def fiftyone_run():

    if dataset_name not in fo.list_datasets():
        print(f"Dataset '{dataset_name}' does not exist in fiftyone database")

    dataset = fo.load_dataset(dataset_name) 

    if not dataset.persistent:
        dataset.persistent = True

    view = dataset.view()
    print(view)

    session = fo.launch_app(dataset)
    session.wait()


if __name__ == "__main__":
    fiftyone_run()
