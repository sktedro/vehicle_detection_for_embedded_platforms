import os
import csv
import cv2
import numpy as np
import pickle

import common


mio_tcd_classes_map = {
    "pedestrian":            common.classes.index("pedestrian"),
    "bicycle":               common.classes.index("bicycle"),
    "motorcycle":            common.classes.index("motorcycle"),
    "car":                   common.classes.index("passenger_car"),
    # "non-motorized_vehicle": common.classes.index("trailer"),
    "non-motorized_vehicle": -1, # Ignore trailers
    "pickup_truck":          common.classes.index("transporter"),
    "work_van":              common.classes.index("transporter"),
    "bus":                   common.classes.index("bus"),
    "articulated_truck":     common.classes.index("truck"),
    "single_unit_truck":     common.classes.index("truck"),
    "motorized_vehicle":     common.classes.index("unknown")
}


def process_mio_tcd():
    """Converts ground truth data of the MIO-TCD dataset's train subset from CSV
    to mmdetection's middle format in a pickle file

    This can take several minutes

    Part of code taken from view_bounding_boxes.py script provided by creators of MIO-TCD

    MIO-TCD format:
    `00000000, pickup_truck, 213, 34, 255, 50` as 
    `a, label, x1, y1, x2, y2`

    or, if the row has 7 values:
    `00000000, pickup_truck, 0.9, 213, 34, 255, 50` as 
    `a, label, score, x1, y1, x2, y2`

    mmdetection format:
    ```json
    [
        {
            'filename': 'a.jpg',
            'width': 1280,
            'height': 720,
            'ann': {
                'bboxes': <np.ndarray> (n, 4) in (x1, y1, x2, y2) order (values are in pixels),
                'labels': <np.ndarray> (n, ),
                'bboxes_ignore': <np.ndarray> (k, 4), (optional field)
                'labels_ignore': <np.ndarray> (k, 4) (optional field)
            }
        },
        ...
    ]
    ```
    """

    # Initialize paths
    # dataset_path = common.datasets["mio-tcd"]["path"]
    dataset_path = os.path.join(common.datasets_path, common.datasets["mio-tcd"]["path"])
    gt_csv_path = os.path.join(dataset_path, "gt_train.csv")
    gt_pickle_path = os.path.join(dataset_path, common.gt_pickle_filename)
    imgs_path = os.path.join(dataset_path, "train")

    # Let's first fetch the data to a dictionary with filenames as keys
    data_dict = {}
    with open(gt_csv_path) as csv_f:
        reader = csv.reader(csv_f, delimiter=',')

        counter = 0
        for row in reader:
            if counter % 1_000 == 0:
                print(f"Processing line {counter}", end="\r")
            counter += 1
            
            # Image name
            img_name = row[0] + ".jpg"

            # Vehicle class (while mapping from MIO-TCD to ours representation)
            cls = row[1]
            cls = mio_tcd_classes_map[cls]

            # This needs to be placed here, unfortunately. Mmdetection displays
            # an error when training if this is placed after the next if clause
            if cls == -1: 
                continue

            # Initialize the object in `data_dict` var if it doesn't exist yet
            if img_name not in data_dict.keys():

                # Height and width of an image
                height, width, _ = cv2.imread(os.path.join(imgs_path, img_name)).shape

                data_dict[img_name] = {
                    "width": int(width),
                    "height": int(height),
                    "ann": {
                        "bboxes": [],
                        "labels": []
                    }
                }
            
            # Bounding box
            if len(row) == 6:
                bbox = [int(val) for val in row[2:]]
            elif len(row) == 7:
                # Skip score field at 2nd index
                bbox = [int(val) for val in row[3:]]
            else:
                raise Exception("Encountered a row with length != 6 and != 7")

            data_dict[img_name]["ann"]["bboxes"].append(bbox)
            data_dict[img_name]["ann"]["labels"].append(cls)

    print("\nData loaded")

    # Convert data_dict to a list and all lists (bboxes and labels) to numpy arrays
    data_list = []
    for key in list(data_dict.keys()):
        val = data_dict[key]
        val["filename"] = os.path.join("train", key)

        # Convert lists of bboxes and labels to arrays
        # Should work if done the same way as labels, but to be sure..:
        val["ann"]["bboxes"] = np.array(
            [np.array(l) for l in val["ann"]["bboxes"]], 
            dtype=np.int16)
        val["ann"]["labels"] = np.array(val["ann"]["labels"], dtype=np.int16)

        data_list.append(val)

    print("Converted to mmdetection's middle format")

    # Write the list to a file
    with open(gt_pickle_path, 'wb') as f:
        pickle.dump(data_list, f, protocol=common.pickle_file_protocol)

    print("Done")

if __name__ == "__main__":
    process_mio_tcd()