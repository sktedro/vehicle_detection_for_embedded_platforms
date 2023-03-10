import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import paths


masked_imgs_rel_dirpath = "imgs_masked/"

gt_unmasked_filenames = {
    "combined": "gt_processed_unmasked.json",
    "train": "gt_processed_train_unmasked.json",
    "val": "gt_processed_val_unmasked.json",
    "test": "gt_processed_test_unmasked.json",
}
gt_filenames = {
    "combined": "gt_processed.json",
    "train": "gt_processed_train.json",
    "val": "gt_processed_val.json",
    "test": "gt_processed_test.json",
}
gt_combined_filenames = {
    "combined": "dataset.json",
    "train": "train.json",
    "val": "val.json",
    "test": "test.json"
}

classes_ids = {
    "bicycle":       1,
    "motorcycle":    2,
    "car":           3,
    "transporter":   4,
    "bus":           5,
    "truck":         6,
    "trailer":       7,
    "unknown":       8,
    "mask":          9,
}

classes_names = {
    1: "bicycle",
    2: "motorcycle",
    3: "car",
    4: "transporter",
    5: "bus",
    6: "truck",
    7: "trailer",
    8: "unknown",
    9: "mask",
}


# First, images with path containing any tag of `data_split_tags` will be split
# to desired subsets (eg. {"test": "MVI"}: any filepath containing "MVI" will be
# in the test subset) Then, remaining data will be split per `data_split`
datasets = {
    # Paths are relative to the dataset_path
    # TODO rename them to rel_dirpath or something
    "mio-tcd": {
        "path": "MIO-TCD/MIO-TCD-Localization/",
        "data_split_tags": {},
        "data_split": {"train": 0.9, "val": 0.05, "test": 0.05},
        "split_randomly": False,
    },
    "aau": {
        "path": "AAU/",
        "data_split_tags": {},
        "data_split": {"train": 1},
        "split_randomly": False,
        "ignored_folders": ["Egensevej-1", "Egensevej-3", "Egensevej-5"], # Bad quality video
    },
    "ndis": {
        "path": "ndis_park/",
        "data_split_tags": {},
        "data_split": {"train": 1},
        "split_randomly": False,
    },
    "mtid": {
        "path": "MTID/",
        "data_split_tags": {},
        "data_split": {"train": 1},
        "split_randomly": False,
        "ignored_images_ranges": {
            "drone": [
                # Images to ignore because they are not annotated (inclusive)
                [1, 31],
                [659, 659],
                [1001, 1318],
                [3301, 3327]
            ],
            "infra": [],
        },
    },
    "visdrone_det": {
        "path": "VisDrone2019-DET-test-dev/",
        "data_split_tags": {},
        "data_split": {"train": 1},
        "split_randomly": False,
    },
    "detrac": {
        "path": "DETRAC/",
        "data_split_tags": {
            "val": ["MVI_40201", "MVI_40244"],
            "test": ["MVI_40204", "MVI_40243"],
        },
        "data_split": {"train": 1},
        "split_randomly": False,
        # "ignored_sequences": [], # No need after reannotation in labelbox
    },
}

for dataset_name in datasets:
    distribution_sum = 0
    for subset in ["train", "val", "test"]:
        distribution_sum += datasets[dataset_name]["data_split"].get(subset, 0)
    assert distribution_sum == 1, f"Data distribution for dataset '{dataset_name}' != 1"


def save_processed(dataset_name, data):
    import json
    from tqdm import tqdm
    from split_dataset_into_subsets import split_dataset

    data = {
        "combined": data
    }

    # Save dataset name to each image
    for img in data["combined"]["images"]:
        img["dataset_name"] = dataset_name

    # Save categories
    data["combined"]["categories"] = []
    for cls_id in list(classes_names.keys()):
        data["combined"]["categories"].append({
            "supercategory": "vehicle",
            "id": cls_id,
            "name": classes_names[cls_id]
        })
        if classes_names[cls_id] == "mask":
            del data["combined"]["categories"][-1]["supercategory"]

    # Add area for each annotation (because mmdetection requires it...)
    # Although it should be the segmentation area, I don't have segmentations
    # so this will have to do...
    # Also add iscrowd to every annotation because pycocotools requires it...
    print("Calcluating areas and adding 'iscrowd' (always 0)")
    for anno in tqdm(data["combined"]["annotations"]):
        anno["area"] = anno["bbox"][2] * anno["bbox"][3]  # area = w * h
        anno["iscrowd"] = 0

    print("Cleaning up the data")
    # Only keep main keys in the data
    for key in list(data["combined"].keys()):
        if key not in ["images", "annotations", "categories", "info", "licenses"]:
            del data["combined"][key]
    # Remove unnecessary key from images
    for img in tqdm(data["combined"]["images"]):
        for key in ["coco_url", "flickr_url", "date_captured"]:
            if key in img:
                del img[key]
    # Remove unnecessary key from annotations
    for anno in tqdm(data["combined"]["annotations"]):
        for key in ["segmentation"]:
            if key in anno:
                del anno[key]
    # Remove annotations with areas being 0, because they could cause problems
    for anno in data["combined"]["annotations"].copy():
        if anno["area"] <= 0:
            data["combined"]["annotations"].remove(anno)

    # Split into train, val, test
    data = split_dataset(dataset_name, data)

    # Save it to files
    print("Saving...")
    dataset_path = os.path.join(
        paths.datasets_dirpath, datasets[dataset_name]["path"])
    for key in ["combined", "train", "val", "test"]:
        gt_filepath = os.path.join(dataset_path, gt_unmasked_filenames[key])
        with open(gt_filepath, 'w') as f:
            json.dump(data[key], f, indent=2)

        print(f"{key}: \t{len(data[key]['images'])} images \t{len(data[key]['annotations'])} annotations")

    print("Remember to run `apply_masks` after processing a dataset, to " +
          "create masked gt files. Do this even for datasets with no masks!")
