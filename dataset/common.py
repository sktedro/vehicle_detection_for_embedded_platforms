import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import paths


masked_imgs_rel_dirpath = "imgs_masked/"

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
        "data_split": {"train": 1},
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
        "data_split_tags": {"test": ["MVI_40191", "MVI_40204", "MVI_40243"],
                            "val": ["MVI_40192", "MVI_40201", "MVI_40244"]},
        "data_split": {"train": 1},
        "split_randomly": False,
        "ignored_sequences": [],
        # "ignored_sequences": [ # List of sequence names to ignore when processing
        #     # Some sequences don't have comments simply because I was lazy to comment the problem
        #     # They were ignored mainly if containing bicycles or motorcycles
        #     "20011", # Contains unmasked cyclists in left corner
        #     "20012", # Contains unmasked cyclists in left corner
        #     "20032", # Contains unmasked cyclists in left corner
        #     "20033", # Unannotated motorcycle
        #     "20034",
        #     "20035", # Badly masked
        #     "20051", # Badly masked
        #     "20061", # Badly annotated
        #     "20062", # Contains unmasked cyclists in left corner
        #     "20063", # Cyclists in left corner
        #     "20064", # Cyclists in left corner
        #     "20065", # Unannotated motorcycle
        #     "39031", # Badly masked + motorcycle
        #     "39051", # Motorcycles
        #     "39211",
        #     "39271", # Motorcycles
        #     "39311", # Motorcycles
        #     "39361", # Bicycles and motorcycles
        #     "39371", # Bicycles
        #     "39401", # Two-wheelers
        #     "39501", # Bicycles
        #     "39511",
        #     "39761", # Bicycles
        #     "39771", # Badly masked
        #     "39781", # Motorcycles
        #     "39801", # Badly masked + two-wheeler
        #     "39811", # Badly masked + two-wheeler
        #     "39821", # Motorcycles
        #     "39851",
        #     "39861", # Bicycle
        #     "39931", # Motorcycles
        #     "40131", # Cyclists
        #     "40141", # Cyclists
        #     "40152", # Bicycle
        #     "40161", # Motorcycle
        #     "40162", # Badly masked
        #     "40171", # Bicycle
        #     "40172", # Badly annotated
        #     "40181", # Two-wheelers
        #     "40191",
        #     "40192", # Badly masked
        #     "40204", # Motorcycle
        #     "40211", # Motorcycle
        #     "40212", # Motorcycle
        #     "40241", # Motorcycle
        #     "40244", # Unannotated bus
        #     "40701",
        #     "40711",
        #     "40712",
        #     "40714",
        #     "40732",
        #     "40742",
        #     "40743",
        #     "40751",
        #     "40752",
        #     "40761", # Bicycles and motorcycles
        #     "40762",
        #     "40763",
        #     "40771",
        #     "40772",
        #     "40773",
        #     "40774",
        #     "40775",
        #     "40792",
        #     "40793",
        #     "40851",
        #     "40852",
        #     "40853",
        #     "40854",
        #     "40855",
        #     "40863",
        #     "40864",
        #     "40871",
        #     "40891",
        #     "40892",
        #     "40901",
        #     "40902",
        #     "40903",
        #     "40904",
        #     "40905",
        #     "40981",
        #     "40991",
        #     "40992",
        #     "63521",
        #     "63525",
        #     "63544",
        #     "63561", # Badly masked/annotated
        #     "63562", # Badly masked/annotated
        #     "63563", # Two-wheeler
        # ],
    },
}

for dataset_name in datasets:
    distribution_sum = 0
    for subset in ["train", "val", "test"]:
        distribution_sum += datasets[dataset_name]["data_split"].get(subset, 0)
    assert distribution_sum == 1, f"Data distribution for dataset '{dataset_name}' != 1"


def split_dataset(dataset_name, data):
    from tqdm import tqdm
    import random
    random.seed(42)

    # Randomly reorder images if desired
    if "split_randomly" in datasets[dataset_name] and datasets[dataset_name]["split_randomly"] == True:
        random.shuffle(data["combined"]["images"])

    # Output - datasets split into train/val/test
    data = {
        "combined": data["combined"],
        "train": {"images": [], "annotations": [], "categories": []},
        "val": {"images": [], "annotations": [], "categories": []},
        "test": {"images": [], "annotations": [], "categories": []}
    }

    # Split data into train/val/test
    images_combined_backup = data["combined"]["images"].copy()

    # Split based on tags (simply remove from "combined" and append to
    # subset
    for img in tqdm(data["combined"]["images"].copy()):
        for subset, tags in datasets[img["dataset_name"]]["data_split_tags"].items():
            for tag in tags:
                if tag in img["file_name"]:
                    data[subset]["images"].append(img)
                    data["combined"]["images"].remove(img)

    # Split based on percentages
    distributed = 0 # Amount of images (from index 0) already distributed into subsets
    for subset in ["train", "val", "test"]:
        amount = int(len(data["combined"]["images"]) * datasets[dataset_name]["data_split"].get(subset, 0))
        data[subset]["images"] += data["combined"]["images"][distributed:distributed+amount]
        distributed += amount

    # Restore "combined"
    data["combined"]["images"] = images_combined_backup

    # Add annotations and categories to splits
    for subset in ["train", "val", "test"]:

        # Save image IDs per subset to img_ids
        img_ids = set()
        for img in data[subset]["images"]:
            img_ids.add(img["id"])

        # For each image, find all its annotations and add them to the
        # same subset
        for anno in data["combined"]["annotations"].copy():
            if anno["image_id"] in img_ids:
                data[subset]["annotations"].append(anno)

        data[subset]["categories"] = data["combined"]["categories"]

    return data


def save_processed(dataset_name, data):
    import json
    from tqdm import tqdm

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
    print("Splitting into train, val and test...")
    data = split_dataset(dataset_name, data)

    # Save it to files
    print("Saving...")
    dataset_path = os.path.join(
        paths.datasets_dirpath, datasets[dataset_name]["path"])
    for key in ["combined", "train", "val", "test"]:
        gt_filepath = os.path.join(dataset_path, gt_filenames[key])
        with open(gt_filepath, 'w') as f:
            json.dump(data[key], f, indent=2)

        print(f"{key}: \t{len(data[key]['images'])} images \t{len(data[key]['annotations'])} annotations")
