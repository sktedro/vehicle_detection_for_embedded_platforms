import os
import cv2
import numpy as np
import pickle
import json
import shutil
import gc

import common

# Classes persent in the dataset: 1, 2, 3, 4, 6, 8, or in text format:
# person, bicycle, car, motorbike, bus, truck
aau_classes_map = {
    1: common.classes.index("pedestrian"),    # person
    2: common.classes.index("bicycle"),       # bicycle
    4: common.classes.index("motorcycle"),    # motorbike
    3: common.classes.index("passenger_car"), # car
    6: common.classes.index("bus"),           # bus
    8: common.classes.index("truck")          # truck
}


def process_aau():
    """Converts ground truth data of the AAU RainSnow dataset from JSON (COCO
    format) to mmdetection's middle format in a pickle file

    AAU dataset (COCO) format (only showing fields of interes):
    ```json
    {
        "images": [
            {
                "id": 0,
                "width": 640,
                "height": 480,
                "file_name": "Egensevej/Egensevej-1/cam1-00055.png"
                }
                ...
            ],
        "annotations": [
            {
                "image_id": 0,
                "category_id": 3,
                "bbox": [
                    402, # Top left x
                    185, # Top left y
                    18, # Width
                    21 # Height
                    ]
                },
                ...
            ]
        "categories": [
            {
                "id": 1,
                "name": "person"
                },
                ...
            ]
    ```

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
    dataset_path = os.path.join(common.datasets_path, common.datasets["aau"]["path"])
    gt_json_path = os.path.join(dataset_path, "aauRainSnow-rgb.json")
    gt_pickle_path = os.path.join(dataset_path, common.gt_pickle_filename)

    # For some reason, Egensevej-5 mask has the wrong filename - fix that:
    if not os.path.exists(os.path.join(dataset_path, "Egensevej", "Egensevej-5-mask.png")):
        dirpath = os.path.join(dataset_path, "Egensevej")
        src = os.path.join(dirpath, "Egensevej-5.png")
        dst = os.path.join(dirpath, "Egensevej-5-mask.png")
        shutil.copy(src, dst)

    # Let's first fetch the data to a dictionary with filenames as keys
    with open(os.path.join(gt_json_path)) as f:
        data = json.loads(f.read())

    data_dict = {}

    # Fetch all annotations first
    for anno in data["annotations"]:

        # Ignore annotations that have negative width or height
        if anno["bbox"][2] < 0 or anno["bbox"][3] < 0:
            continue

        img_id = anno["image_id"]

        # If the image is not yet in data_dict, initialize it
        if img_id not in data_dict.keys():
            data_dict[img_id] = {
                "ann": {
                    "bboxes": [],
                    "labels": []
                }
            }

        # Append annotation (class (label) and bbox)
        data_dict[img_id]["ann"]["labels"].append(aau_classes_map[anno["category_id"]])

        # Convert bbox from [x1 y1 w h] to [x1 y1 x2 y2]
        bbox = anno["bbox"]
        bbox[2] = bbox[0] + bbox[2]
        bbox[3] = bbox[1] + bbox[3]
        data_dict[img_id]["ann"]["bboxes"].append(bbox)

    # Append info about the images
    # Image: id, width, height, file_name
    for image in data["images"]:
        img_id = image["id"]

        # In case there is an image with no annotations...:
        if img_id not in data_dict.keys():
            data_dict[img_id] = {
                "ann": {
                    "bboxes": [],
                    "labels": []
                }
            }

        data_dict[img_id]["width"] = image["width"]
        data_dict[img_id]["height"] = image["height"]
        data_dict[img_id]["filename"] = image["file_name"]

    # We can free data from memory
    del data
    gc.collect()

    print("Data loaded")

    # Remove ignored folders
    for img_id in data_dict.copy():
        for ignore_str in common.datasets["aau"]["ignored_folders"]:
            if ignore_str in data_dict[img_id]["filename"]:
                del data_dict[img_id]
                break

    # Create a directory for combined images (delete it first if exists)
    combined_imgs_rel_path = "imgs_combined"
    combined_imgs_path = os.path.join(dataset_path, combined_imgs_rel_path)
    if os.path.exists(combined_imgs_path):
        shutil.rmtree(combined_imgs_path)
    os.mkdir(combined_imgs_path)

    # Copy all images to a separate folder while applying detection masks (so
    # that the image only contains annotated vehicles) and update the paths
    for img_id in data_dict:
        old_img_filepath = os.path.join(dataset_path, data_dict[img_id]["filename"])
        new_img_rel_path = os.path.join(combined_imgs_rel_path, str(img_id).zfill(9) + ".jpg")

        cam_dirpath = os.path.dirname(old_img_filepath) # .../Egensevej-1
        location_dirpath = os.path.dirname(cam_dirpath) # .../Evensevej

        # Mask filename example: Evensevej-1-mask.jpg in the same directory as
        # Egensevej-1/ directory
        mask_filename = os.path.basename(cam_dirpath) + "-mask.png"
        mask_filepath = os.path.join(location_dirpath, mask_filename)

        frame = cv2.imread(old_img_filepath)
        mask = cv2.imread(mask_filepath)
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        new_frame = cv2.bitwise_and(frame, frame, mask=mask) # Apply mask

        cv2.imwrite(os.path.join(dataset_path, new_img_rel_path), new_frame)

        data_dict[img_id]["filename"] = new_img_rel_path

    print(f"All images processed and combined to {combined_imgs_path}")

    # Convert data_dict to a list and all lists (bboxes and labels) to numpy arrays
    data_list = []
    for key in list(data_dict.keys()):
        val = data_dict[key]

        # Convert lists of bboxes and labels to arrays
        # Should work if done the same way as labels, but to be sure..:
        val["ann"]["bboxes"] = np.array(
            [np.array(l, dtype=np.int16) for l in val["ann"]["bboxes"]], 
            dtype=np.int16)
        val["ann"]["labels"] = np.array(val["ann"]["labels"], dtype=np.int16)

        data_list.append(val)
    
    print("Converted to mmdetection's middle format")

    # Write the list to a file
    with open(gt_pickle_path, 'wb') as f:
        pickle.dump(data_list, f, protocol=common.pickle_file_protocol)

    print("Done")

if __name__ == "__main__":
    process_aau()