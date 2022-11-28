import os
from pickle import HIGHEST_PROTOCOL, DEFAULT_PROTOCOL

# Paths: (absolute path recommended)
datasets_path = "/home/tedro/Downloads/datasets/"

gt_pickle_filename = "gt.pickle"

dataset_pickle_filepath = os.path.join(datasets_path, "dataset.pickle")
train_pickle_filepath = os.path.join(datasets_path, "train.pickle")
val_pickle_filepath = os.path.join(datasets_path, "val.pickle")
test_pickle_filepath = os.path.join(datasets_path, "test.pickle")
pickle_file_protocol = HIGHEST_PROTOCOL
# pickle_file_protocol = 0

datasets = {
    # paths are relative to the dataset_path
    "mio-tcd": {
        "path": "MIO-TCD/MIO-TCD-Localization/"
    },
    "aau": {
        "path": "AAU/",
        "ignored_folders": ["Egensevej-1", "Egensevej-3", "Egensevej-5"]
    },
    "ndis": {
        "path": "ndis_park/",
        "ignored_images": [ # Ones that have pedestrians
            1, 6, 8, 9, 
            17, 18, 21, 25, 29, 30, 31, 37, 38, 40, 47, 51, 52, 56, 
            61, 67, 68, 70, 72, 81, 82, 84, 85, 87, 88, 89, 95,
            100, 104, 107, 108, 113, 119, 127, 128, 134, 140
        ]
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
        "frame_step": 3 # If set to 10, only each 10th frame will be taken
    }
}

classes = [
    "pedestrian", 
    "bicycle", 
    "motorcycle", 
    "passenger_car",
    "transporter",
    "bus",
    "truck",
    "unknown"
]
classes_dict = {
    0: "pedestrian",
    1: "bicycle", 
    2: "motorcycle", 
    3: "passenger_car",
    4: "transporter",
    5: "bus",
    6: "truck",
    7: "unknown"
}