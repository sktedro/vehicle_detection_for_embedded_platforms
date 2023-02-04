import os
import csv
import cv2
import numpy as np
import pickle
from tqdm import tqdm

# The script should be importable but also executable from the terminal...
if __name__ == '__main__':
    import common
else:
    from . import common

"""Notes:
Coordinates are in pixels
"""


visdrone_det_classes_map = {
    "0":  -1, # Ignore "ignored regions"
    "1":  common.classes.index("pedestrian"),
    "2":  -1, # Ignore "people". Otherwise, person sitting on a motorcycle is annotated, too
    "3":  common.classes.index("bicycle"),
    "4":  common.classes.index("passenger_car"),
    "5":  common.classes.index("transporter"),
    "6":  common.classes.index("truck"),
    "7":  common.classes.index("unknown"), # Tricycle
    "8":  common.classes.index("unknown"), # Awning-tricycle
    "9":  common.classes.index("bus"),
    "10": common.classes.index("motorcycle"), # "motor" is a motorcycle
    "11": -1 # Ignore "others"
}


def process_visdrone_det():
    """Converts ground truth data of the VisDrone DET dataset's train subset
    from CSV to mmdetection's middle format in a pickle file

    This can take several minutes

    VisDrone DET format:
    <bbox_left>,<bbox_top>,<bbox_width>,<bbox_height>,<score>,<object_category>,<truncation>,<occlusion>
    <bbox_left>    The x coordinate of the top-left corner of the predicted bounding box
    <bbox_top>     The y coordinate of the top-left corner of the predicted object bounding box
    <bbox_width>   The width in pixels of the predicted object bounding box
    <bbox_height>  The height in pixels of the predicted object bounding box
    <score>	       The score in the DETECTION file indicates the confidence of the predicted bounding box enclosing 
                   an object instance.
                   The score in GROUNDTRUTH file is set to 1 or 0. 1 indicates the bounding box is considered in evaluation, 
                   while 0 indicates the bounding box will be ignored.
    <object_category>  The object category indicates the type of annotated object, (i.e., ignored regions(0), pedestrian(1), 
                       people(2), bicycle(3), car(4), van(5), truck(6), tricycle(7), awning-tricycle(8), bus(9), motor(10), 
                       others(11))
    <truncation>  The score in the DETECTION result file should be set to the constant -1.
                  The score in the GROUNDTRUTH file indicates the degree of object parts appears outside a frame 
                  (i.e., no truncation = 0 (truncation ratio 0%), and partial truncation = 1 (truncation ratio 1% ~ 50%)).
    <occlusion>  The score in the DETECTION file should be set to the constant -1.
                 The score in the GROUNDTRUTH file indicates the fraction of objects being occluded (i.e., no occlusion = 0 
                 (occlusion ratio 0%), partial occlusion = 1 (occlusion ratio 1% ~ 50%), and heavy occlusion = 2 
                 (occlusion ratio 50% ~ 100%)).

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
    dataset_path = os.path.join(common.datasets_path, common.datasets["visdrone_det"]["path"])
    gts_path = os.path.join(dataset_path, "annotations")
    gt_pickle_path = os.path.join(dataset_path, common.gt_pickle_filename)
    imgs_path = os.path.join(dataset_path, "images")

    gts_files_list = os.listdir(gts_path)
    imgs_files_list = os.listdir(imgs_path)

    # Let's first fetch the data to a dictionary with filenames as keys
    data_dict = {}

    total = len(gts_files_list)
    print("Loading and processing annotations")
    for gt_file in tqdm(gts_files_list):

        img_filename = gt_file.split(".")[0] + ".jpg"

        # Check that image exists
        # assert img_filename in imgs_files_list
        if img_filename not in imgs_files_list:
            print(f"Could not find image \"{img_filename}\". Ignoring")
            continue

        gt_filepath = os.path.join(gts_path, os.path.basename(gt_file))
        with open(gt_filepath) as csv_f:
            reader = csv.reader(csv_f, delimiter=',')

            for row in reader:

                bbox_left, bbox_top, w, h, _, cls, _, _ = row

                # Vehicle class (while mapping from MIO-TCD to ours representation)
                cls = visdrone_det_classes_map[cls]

                # Initialize the object in `data_dict` var if it doesn't exist yet
                if img_filename not in data_dict.keys():

                    # Height and width of an image
                    height, width, _ = cv2.imread(os.path.join(imgs_path, img_filename)).shape

                    data_dict[img_filename] = {
                        "width": int(width),
                        "height": int(height),
                        "ann": {
                            "bboxes": [],
                            "labels": []
                        }
                    }
                
                # If class is -1, ignore this annotation
                if cls != -1: 

                    # Bounding box
                    bbox_left = int(bbox_left)
                    bbox_top = int(bbox_top)
                    w = int(w)
                    h = int(h)
                    bbox = [bbox_left, bbox_top, bbox_left + w, bbox_top + h]
                
                    data_dict[img_filename]["ann"]["bboxes"].append(bbox)
                    data_dict[img_filename]["ann"]["labels"].append(cls)

    # Convert data_dict to a list and all lists (bboxes and labels) to numpy arrays
    data_list = []
    print("Converting...")
    for key in tqdm(list(data_dict.keys())):
        val = data_dict[key]
        val["filename"] = os.path.join("images", key)

        # Convert lists of bboxes and labels to arrays
        # Should work if done the same way as labels, but to be sure..:
        val["ann"]["bboxes"] = np.array(
            [np.array(l) for l in val["ann"]["bboxes"]], 
            dtype=np.int16)
        val["ann"]["labels"] = np.array(val["ann"]["labels"], dtype=np.int16)

        data_list.append(val)

    print(f"Images: {len(data_list)}")
    annotations = sum([len(img["ann"]["labels"]) for img in data_list])
    print(f"Annotations: {annotations}")

    # Write the list to a file
    with open(gt_pickle_path, 'wb') as f:
        pickle.dump(data_list, f, protocol=common.pickle_file_protocol)

    print(f"Saved to {gt_pickle_path}")

if __name__ == "__main__":
    process_visdrone_det()