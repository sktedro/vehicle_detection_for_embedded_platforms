import os
import numpy as np
import pickle
import json
import shutil
import gc
from tqdm import tqdm

# The script should be importable but also executable from the terminal...
if __name__ == '__main__':
    import common
else:
    from . import common

# Dataset only contains one class: car (3)
# It also contains unannotated people, which we don't care about since we don't
# detect them.
# Also contains trailers, trucks and transporters. These are mapped manually by
# setClass() calls in the code
ndis_classes_map = {
    3: common.classes.index("passenger_car"), # car
}


def process_ndis():
    """Converts ground truth data of the NDISPark dataset from JSON (COCO
    format) to mmdetection's middle format in a pickle file

    NDIS (COCO) dataset format (only showing fields of interest):
    ```json
    {
        "images": [
            {
                "id": 0,
                "width": 2400,
                "height": 908,
                "file_name": "64_1537102819.jpg"
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
    dataset_path = os.path.join(common.datasets_path, common.datasets["ndis"]["path"])
    gt_json_path = {
        "train": os.path.join(dataset_path, "train/train_coco_annotations.json"),
        "val": os.path.join(dataset_path, "validation/val_coco_annotations.json")
    }
    imgs_path = {
        "train": os.path.join(dataset_path, "train/imgs"),
        "val": os.path.join(dataset_path, "validation/imgs")
    }
    gt_pickle_path = os.path.join(dataset_path, common.gt_pickle_filename)

    # Create a directory for combined images (delete it first if exists)
    combined_imgs_rel_path = "imgs_combined"
    combined_imgs_path = os.path.join(dataset_path, combined_imgs_rel_path)
    if os.path.exists(combined_imgs_path):
        shutil.rmtree(combined_imgs_path)
    os.mkdir(combined_imgs_path)

    # Read the data while copying all images to combined/ directory (and
    # ignoring images that should be ignored)
    data_dict = {}
    print(f"Loading data and copying images to {combined_imgs_path}")
    for key in gt_json_path:

        try:
            max_id = max(list(data_dict.keys()))
        except:
            max_id = -1

        with open(gt_json_path[key]) as f:
            data = json.loads(f.read())

            # Fetch all annotations first and save them
            print(f"Processing annotations in {key}")
            for anno in tqdm(data["annotations"]):
                new_img_id = anno["image_id"] + max_id + 1
                if new_img_id in common.datasets["ndis"]["ignored_images"]:
                    continue

                # If the image is not yet in data_dict, initialize it
                if new_img_id not in data_dict.keys():
                    data_dict[new_img_id] = {
                        "ann": {
                            "bboxes": [],
                            "labels": []
                        }
                    }

                # Append annotation (class (label) and bbox)
                data_dict[new_img_id]["ann"]["labels"].append(ndis_classes_map[anno["category_id"]])

                # Convert bbox from [x1 y1 w h] to [x1 y1 x2 y2]
                bbox = anno["bbox"]
                bbox[2] = bbox[0] + bbox[2]
                bbox[3] = bbox[1] + bbox[3]
                data_dict[new_img_id]["ann"]["bboxes"].append(bbox)

            # Append info about the images while copying the images to combined/
            # Image: id, width, height, file_name
            print(f"Processing images in {key}")
            for image in tqdm(data["images"]):
                new_img_id = image["id"] + max_id + 1
                if new_img_id in common.datasets["ndis"]["ignored_images"]:
                    continue

                # In case there is an image with no annotations...:
                if new_img_id not in data_dict.keys():
                    data_dict[new_img_id] = {
                        "ann": {
                            "bboxes": [],
                            "labels": []
                        }
                    }

                data_dict[new_img_id]["width"] = image["width"]
                data_dict[new_img_id]["height"] = image["height"]

                # Update the filepath
                new_img_rel_filepath = os.path.join(combined_imgs_rel_path, str(new_img_id).zfill(9) + ".jpg")
                data_dict[new_img_id]["filename"] = new_img_rel_filepath

                # Copy image to combined/
                old_img_filepath = os.path.join(imgs_path[key], image["file_name"])
                new_img_filepath = os.path.join(dataset_path, new_img_rel_filepath)
                shutil.copy(old_img_filepath, new_img_filepath)

    # We can free data from memory
    del data
    gc.collect()

    def setClass(img_id, bbox_index, cls):
        if type(bbox_index) in [list, tuple]:
            for i in range(len(bbox_index)):
                setClass(img_id, bbox_index[i], cls)
            return

        if type(cls) == int:
            data_dict[img_id]["ann"]["labels"][bbox_index] = cls
        else:
            try:
                cls = common.classes.index(cls)
                data_dict[img_id]["ann"]["labels"][bbox_index] = cls
            except ValueError:
                print(f"Manual class fix failed at img_id {img_id}, bbox_index{bbox_index}")
    
    # Be careful! This function operates with indices! Only call once per image
    # and only after all setClass methods
    def delBbox(img_id, bbox_index):
        del data_dict[img_id]["ann"]["labels"][bbox_index]
        del data_dict[img_id]["ann"]["bboxes"][bbox_index]

    print("Fixing annotations...")

    # delBbox() was used to delete bounding boxes of trailers. Now that I'm
    # adding trailers back (since I realized I want to detect them), I commented
    # out those calls (all delBbox() calls were because of trailers)
    setClass(2, 15, "transporter")
    setClass(3, 2, "transporter")
    setClass(4, [5, 4], "transporter")
    setClass(5, 2, "transporter")
    setClass(8, 4, "transporter")
    setClass(9, [13, 14, 29], "transporter")
    setClass(11, 5, "transporter")
    setClass(11, 0, "trailer")
    # delBbox(11, 0)
    setClass(12, 2, "truck")
    setClass(12, 3, "truck")
    setClass(14, 6, "transporter")
    setClass(15, 0, "transporter")
    setClass(16, 6, "transporter")
    setClass(18, 59, "trailer")
    setClass(25, 36, "transporter")
    setClass(27, 11, "transporter")
    setClass(28, 0, "transporter")
    setClass(30, [0, 6], "transporter")
    setClass(32, [16, 35, 39, 25, 32], "transporter")
    setClass(35, [4, 5], "transporter")
    setClass(36, 3, "transporter")
    setClass(39, [19, 18], "transporter")
    setClass(40, [25, 29], "transporter")
    setClass(41, 2, "transporter")
    setClass(42, 1, "transporter")
    setClass(44, 8, "transporter")
    setClass(46, 6, "transporter")
    setClass(47, [6, 17], "transporter")
    setClass(50, 3, "transporter")
    # delBbox(53, 60)
    setClass(53, 60, "trailer")
    setClass(54, 0, "transporter")
    setClass(55, 53, "transporter")
    setClass(58, 4, "transporter")
    setClass(59, 46, "transporter")
    # delBbox(60, 1)
    setClass(60, 1, "trailer")
    setClass(62, 10, "transporter")
    # delBbox(64, 71)
    setClass(64, 71, "trailer")
    setClass(66, 0, "transporter")
    setClass(68, 1, "trailer")
    setClass(69, 4, "transporter")
    setClass(73, [36, 45, 10], "transporter")
    setClass(74, 4, "transporter")
    setClass(75, 35, "transporter")
    # delBbox(75, 64)
    setClass(75, 65, "trailer")
    setClass(77, 1, "transporter")
    setClass(73, 2, "transporter")
    setClass(81, [4, 21, 24, 40], "transporter")
    setClass(82, 7, "transporter")
    setClass(83, 2, "transporter")
    setClass(85, 1, "trailer")
    setClass(87, 7, "transporter")
    setClass(88, 16, "transporter")
    setClass(89, 69, "trailer")
    setClass(90, 2, "transporter")
    setClass(93, [41, 28, 37, 42, 16], "transporter")
    setClass(94, [1, 26], "transporter")
    setClass(96, 1, "transporter")
    setClass(98, [9, 12, 31], "transporter")
    setClass(103, 15, "transporter")
    setClass(106, 1, "transporter")
    setClass(107, 42, "transporter")
    setClass(108, 4, "transporter")
    setClass(109, 5, "transporter")
    setClass(115, 45, "transporter")
    setClass(117, 3, "transporter")
    setClass(118, 0, "transporter")
    setClass(120, [6, 8], "transporter")
    setClass(122, 1, "transporter")
    setClass(123, 0, "transporter")
    setClass(125, 11, "transporter")
    setClass(127, 1, "trailer")
    setClass(128, [3, 14], "transporter")
    setClass(129, [33, 34, 24, 3], "transporter")
    setClass(131, [1, 3], "transporter")
    setClass(133, 6, "transporter")
    setClass(136, [6, 1], "transporter")
    setClass(139, [42, 1], "transporter")
    setClass(140, [5, 6], "transporter")
    
    # Convert data_dict to a list and all lists (bboxes and labels) to numpy arrays
    data_list = []
    print("Converting...")
    for key in tqdm(list(data_dict.keys())):
        val = data_dict[key]

        # Convert lists of bboxes and labels to arrays
        # Should work if done the same way as labels, but to be sure..:
        val["ann"]["bboxes"] = np.array(
            [np.array(l, dtype=np.int16) for l in val["ann"]["bboxes"]], 
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
    process_ndis()