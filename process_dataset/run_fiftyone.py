import fiftyone as fo
import common
from common import datasets_path
from tqdm import tqdm


def run_fiftyone(delete=False):
    dataset_name = "dataset"

    if dataset_name in fo.list_datasets():
        dataset = fo.load_dataset(dataset_name) 

    else:
        dataset = fo.Dataset.from_dir(
            name=dataset_name,
            data_path=common.datasets_path,
            labels_path=common.dataset_coco_filepath,
            dataset_type=fo.types.COCODetectionDataset
        )

        print("Updating tags")
        for sample in tqdm(dataset):
            ds_name = sample["filepath"].split(datasets_path)[1].split("/")[0]
            if ds_name not in sample.tags:
                sample.tags.append(ds_name)
                sample.save()

    if not dataset.persistent:
        dataset.persistent = True

    if delete:
        dataset.delete()
        run_fiftyone()
        return

    view = dataset.view()
    print(view)

    session = fo.launch_app(dataset)
    session.wait()


if __name__ == "__main__":
    run_fiftyone(delete=False)
