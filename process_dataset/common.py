import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import paths


gt_filename = "gt_processed.json"

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
    print("Calcluating areas")
    for anno in tqdm(data["annotations"]):
        anno["area"] = anno["bbox"][2] * anno["bbox"][3] # area = w * h

    # Clear the data further
    print("Cleaning up the data")
    for img in tqdm(data["images"]):
        for key in list(img.keys()):
            if key not in ["id", "width", "height", "file_name", "license"]:
                del img[key]
    for anno in tqdm(data["annotations"]):
        for key in list(anno.keys()):
            if key not in ["id", "image_id", "category_id", "area", "bbox"]:
                del anno[key]
    for key in list(data.keys()):
        if key not in ["images", "annotations", "categories", "info", "licenses"]:
            del data[key]

    print(f"Images: {len(data['images'])}")
    print(f"Annotations: {len(data['annotations'])}")

    # Write it to a file
    dataset_path = os.path.join(paths.datasets_dirpath, datasets[dataset_name]["path"])
    gt_filepath = os.path.join(dataset_path, gt_filename)
    with open(gt_filepath, 'w') as f:
        f.write(json_dumps(data, indent=2))

    print(f"Saved to {gt_filepath}")
