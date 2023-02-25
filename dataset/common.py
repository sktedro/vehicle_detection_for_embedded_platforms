import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import paths


# Provide data split values for train, val and testing data as percentage / 100
data_distribution = {
    "train": 0.9,
    "val": 0.05,
    "test": 0.05
}
# Select if you want your splits to be continuous vs random
# Eg. of continuous: train will contain 0.jpg, 1.jpg, 2.jpg, ...
random_data_distribution = True
assert sum(data_distribution[key] for key in data_distribution.keys()) == 1
gt_filenames = {
    "combined": "gt_processed.json",
    "train": "gt_processed_train.json",
    "val": "gt_processed_val.json",
    "test": "gt_processed_test.json",
}
gt_combined_filenames = { # TODO or filepaths?? as it was before
    "combined": "dataset.json",
    "train": "train.json",
    "val": "val.json",
    "test": "test.json"
}

# TODO deprecate this:
gt_filename = "gt_processed.json"
gt_train_filename = "gt_processed_train.json"
gt_val_filename = "gt_processed_val.json"
gt_test_filename = "gt_processed_test.json"
dataset_filepath = os.path.join(paths.datasets_dirpath, "dataset.json")
dataset_train_filepath = os.path.join(paths.datasets_dirpath, "train.json")
dataset_val_filepath = os.path.join(paths.datasets_dirpath, "val.json")
dataset_test_filepath = os.path.join(paths.datasets_dirpath, "test.json")

classes_ids = {
    "bicycle":       1,
    "motorcycle":    2,
    "car":           3,
    "transporter":   4,
    "bus":           5,
    "truck":         6,
    "trailer":       7,
    "unknown":       8
}

classes_names = {
    1: "bicycle",
    2: "motorcycle",
    3: "car",
    4: "transporter",
    5: "bus",
    6: "truck",
    7: "trailer",
    8: "unknown"
}

datasets = {
    # Paths are relative to the dataset_path
    # TODO rename them to rel_dirpath or something
    "mio-tcd": {
        "path": "MIO-TCD/MIO-TCD-Localization/"
    },
    "aau": {
        "path": "AAU/",
        "ignored_folders": ["Egensevej-1", "Egensevej-3", "Egensevej-5"] # Bad quality video
    },
    "ndis": {
        "path": "ndis_park/"
    },
    "mtid": {
        "path": "MTID/",
        "ignored_images_ranges": {
            "drone": [
                # Images to ignore because they are not annotated
                [1, 31], [659, 659], [1001, 1318], [3301, 3327]
            ],
            "infra": []
        }
    },
    "visdrone_det": {
        "path": "VisDrone2019-DET-test-dev/"
    },
    "detrac": {
        "path": "DETRAC/",
        "ignored_sequences": [ # List of sequence names to ignore when processing
            # Some sequences don't have comments simply because I was lazy to comment the problem
            # They were ignored mainly if containing bicycles or motorcycles
            "20011", # Contains unmasked cyclists in left corner
            "20012", # Contains unmasked cyclists in left corner
            "20032", # Contains unmasked cyclists in left corner
            "20033", # Unannotated motorcycle
            "20034",
            "20035", # Badly masked
            "20051", # Badly masked
            "20061", # Badly annotated
            "20062", # Contains unmasked cyclists in left corner
            "20063", # Cyclists in left corner
            "20064", # Cyclists in left corner
            "20065", # Unannotated motorcycle
            "39031", # Badly masked + motorcycle
            "39051", # Motorcycles
            "39211",
            "39271", # Motorcycles
            "39311", # Motorcycles
            "39361", # Bicycles and motorcycles
            "39371", # Bicycles
            "39401", # Two-wheelers
            "39501", # Bicycles
            "39511",
            "39761", # Bicycles
            "39771", # Badly masked
            "39781", # Motorcycles
            "39801", # Badly masked + two-wheeler
            "39811", # Badly masked + two-wheeler
            "39821", # Motorcycles
            "39851",
            "39861", # Bicycle
            "39931", # Motorcycles
            "40131", # Cyclists
            "40141", # Cyclists
            "40152", # Bicycle
            "40161", # Motorcycle
            "40162", # Badly masked
            "40171", # Bicycle
            "40172", # Badly annotated
            "40181", # Two-wheelers
            "40191",
            "40192", # Badly masked
            "40204", # Motorcycle
            "40211", # Motorcycle
            "40212", # Motorcycle
            "40241", # Motorcycle
            "40244", # Unannotated bus
            "40701",
            "40711",
            "40712",
            "40714",
            "40732",
            "40742",
            "40743",
            "40751",
            "40752",
            "40761", # Bicycles and motorcycles
            "40762",
            "40763",
            "40771",
            "40772",
            "40773",
            "40774",
            "40775",
            "40792",
            "40793",
            "40851",
            "40852",
            "40853",
            "40854",
            "40855",
            "40863",
            "40864",
            "40871",
            "40891",
            "40892",
            "40901",
            "40902",
            "40903",
            "40904",
            "40905",
            "40981",
            "40991",
            "40992",
            "63521",
            "63525",
            "63544",
            "63561", # Badly masked/annotated
            "63562", # Badly masked/annotated
            "63563", # Two-wheeler
        ]
    },
}


def split_dataset(dataset_name, data):
    import random
    random.seed(42)

    # Output - datasets split into train/val/test
    data_split = {
        "train": {"images": [], "annotations": [], "categories": []},
        "val": {"images": [], "annotations": [], "categories": []},
        "test": {"images": [], "annotations": [], "categories": []}
    }

    # Randomly reorder images if desired
    if random_data_distribution:
        random.shuffle(data["images"])

    # Save dataset name to each image
    for img in data["images"]:
        img["dataset_name"] = dataset_name

    # Split data into train/val/test
    train_val_split_index = int(len(data["images"]) * data_distribution["train"])
    val_test_split_index = int(len(data["images"]) * (data_distribution["train"] + data_distribution["val"]))
    data_split["train"]["images"] = data["images"][:train_val_split_index]
    data_split["val"]["images"]   = data["images"][train_val_split_index:val_test_split_index]
    data_split["test"]["images"]  = data["images"][val_test_split_index:]

    # Add annotations and update their img IDs
    for subset in ["train", "val", "test"]:

        # Save image IDs per subset to img_ids
        img_ids = set()
        for img in data_split[subset]["images"]:
            img_ids.add(img["id"])

        # For each image, find all its annotations and add them to the
        # same subset
        for anno in data["annotations"]:
            if anno["image_id"] in img_ids:
                data_split[subset]["annotations"].append(anno)

        data_split[subset]["categories"] = data["categories"]

    return data_split

def save_processed(dataset_name, data):
    from tqdm import tqdm
    from json import dumps as json_dumps

    # Save categories
    data["categories"] = []
    for cls_id in list(classes_names.keys()):
        data["categories"].append({
            "supercategory": "vehicle",
            "id": cls_id,
            "name": classes_names[cls_id]
        })

    # Add area for each annotation (because mmdetection requires it...)
    # Although it should be the segmentation area, I don't have segmentations
    # so this will have to do...
    # Also add iscrowd to every annotation because pycocotools requires it...
    print("Calcluating areas and adding 'iscrowd' (always 0)")
    for anno in tqdm(data["annotations"]):
        anno["area"] = anno["bbox"][2] * anno["bbox"][3] # area = w * h
        anno["iscrowd"] = 0

    # Remove annotations with areas being 0, because they could cause problems
    for anno in data["annotations"].copy():
        if anno["area"] <= 0:
            data["annotations"].remove(anno)

    # Clear the data further
    print("Cleaning up the data")
    for img in tqdm(data["images"]):
        for key in list(img.keys()):
            if key not in ["dataset_name", "id", "width", "height", "file_name", "license"]:
                del img[key]
    for anno in tqdm(data["annotations"]):
        for key in list(anno.keys()):
            if key not in ["id", "image_id", "category_id", "bbox", "area", "iscrowd"]:
                del anno[key]
    for key in list(data.keys()):
        if key not in ["images", "annotations", "categories", "info", "licenses"]:
            del data[key]

    # Split into train, val, test
    print("Splitting into train, val and test...")
    data_split = split_dataset(dataset_name, data)

    # Write it to a file
    print("Saving...")
    datasets_path = os.path.join(paths.datasets_dirpath, datasets[dataset_name]["path"])
    gt_filepath = os.path.join(datasets_path, gt_filenames["combined"])
    with open(gt_filepath, 'w') as f:
        f.write(json_dumps(data, indent=2))
    for subset in ["train", "val", "test"]:
        gt_filepath = os.path.join(datasets_path, gt_filenames[subset])
        with open(gt_filepath, 'w') as f:
            f.write(json_dumps(data_split[subset], indent=2))

    # Print amounts of images and annotations
    for subset in ["train", "val", "test"]:
        print(f"{subset}: \t{len(data_split[subset]['images'])} images \t{len(data_split[subset]['annotations'])} annotations")
    print(f"Total: \t{len(data['images'])} images \t{len(data['annotations'])} annotations")
