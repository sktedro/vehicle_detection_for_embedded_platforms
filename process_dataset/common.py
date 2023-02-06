from pickle import HIGHEST_PROTOCOL
import os

# Paths: (absolute path recommended)
# datasets_path = "/home/tedro/Downloads/datasets/"
datasets_path = "/Users/z004ktej/Downloads/datasets/"
split_datasets_path = os.path.join(datasets_path, "datasets_gt/")

assert os.path.exists(datasets_path)

gt_pickle_filename = "gt.pickle"

dataset_pickle_filepath = os.path.join(datasets_path, "dataset.pickle")
train_pickle_filepath = os.path.join(datasets_path, "train.pickle")
val_pickle_filepath = os.path.join(datasets_path, "val.pickle")
test_pickle_filepath = os.path.join(datasets_path, "test.pickle")
pickle_file_protocol = HIGHEST_PROTOCOL
# pickle_file_protocol = 0

dataset_coco_filepath = os.path.join(datasets_path, "dataset_coco.json")

datasets = {
    # paths are relative to the dataset_path
    "mio-tcd": {
        "path": "MIO-TCD/MIO-TCD-Localization/"
    },
    "aau": {
        "path": "AAU/",
        "ignored_folders": ["Egensevej-1", "Egensevej-3", "Egensevej-5"] # Bad quality video
    },
    "ndis": {
        "path": "ndis_park/",
        "ignored_images": []
    },
    "mtid": {
        "path": "MTID/",
        "ignored_images_ranges": {
            "drone": [
                # Images to ignore because they are not annotated
                [1, 31], [659, 659], [1001, 1318], [3301, 3327]
            ],
            "infra": []
        },
        "frame_step": 1 # If set to 10, only each 10th frame will be taken
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
    }
    # "coco": {
    #     "path": "coco/"
    # },
}

classes = [
    "bicycle", 
    "motorcycle", 
    "passenger_car",
    "transporter",
    "bus",
    "truck",
    "trailer",
    "unknown"
]

classes_dict = {
    0: "bicycle", 
    1: "motorcycle", 
    2: "passenger_car",
    3: "transporter",
    4: "bus",
    5: "truck",
    6: "trailer",
    7: "unknown"
}
