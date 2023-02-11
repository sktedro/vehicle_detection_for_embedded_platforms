import os
import json
import cv2
import numpy as np
import pickle
from tqdm import tqdm

# The script should be importable but also executable from the terminal...
if __name__ == '__main__':
    import common
else:
    from . import common


# TODO OUTDATED (probably the whole file)
coco_classes_map = { # TODO
    "person":      common.classes.index("pedestrian"),
    "bicycle":     -2, # If an image has this class, ignore whole image
    "motorcycle":  -2, # If an image has this class, ignore whole image
    "car":         common.classes.index("passenger_car"),
    "bus":         common.classes.index("bus"),
    "truck":       common.classes.index("truck"),
}


def process_coco():
    """Converts ground truth data of the COCO dataset from COCO format to
    mmdetection's middle format in a pickle file

    This can take several minutes

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
    dataset_path = os.path.join(common.datasets_dirpath, common.datasets["coco"]["path"])
    gt_pickle_path = os.path.join(dataset_path, common.gt_pickle_filename)
    annotations_paths = {
        "train": os.path.join(dataset_path, "annotations", "instances_train2017.json"),
        "val": os.path.join(dataset_path, "annotations", "instances_val2017.json")
    }
    imgs_dirnames = {
        "train": "train2017",
        "val": "val2017"
    }
    imgs_paths = {
        "train": os.path.join(dataset_path, imgs_dirnames["train"]),
        "val": os.path.join(dataset_path, imgs_dirnames["val"]),
    }
    assert os.path.exists(annotations_paths["train"])
    assert os.path.exists(annotations_paths["val"])
    assert os.path.exists(imgs_paths["train"])
    assert os.path.exists(imgs_paths["val"])

    # Let's first fetch the data to a dictionary with IDs as keys
    data_list = []
    strictly_ignored_counter = 0
    no_anno_counter = 0
    no_vehicle_counter = 0
    for key in ["train", "val"]:

        data_dict = {}
        print(f"Opening annotation file of the {key} subset")
        with open(annotations_paths[key]) as f:

            data = json.loads(f.read())

            # Get classes
            classes = {}
            for cls in data["categories"]:
                classes[cls["id"]] = cls["name"]

            print(f"Processing images of the {key} subset")
            for img in tqdm(data["images"]):
                assert img["id"] not in data_dict.keys() # TODO remove if ok
                data_dict[img["id"]] = {
                    "width": img["width"],
                    "height": img["height"],
                    "filename": img["file_name"],
                    "ann": {
                        "bboxes": [],
                        "labels": [],
                    }
                }
            
            print(f"Processing annotations of the {key} subset")
            for anno in tqdm(data["annotations"]):
                img_id = anno["image_id"]

                try:
                    cls_id = anno["category_id"]
                    cls_coco_name = classes[cls_id]
                    cls = coco_classes_map[cls_coco_name]

                # If there is no rule to map the coco class to our class, ignore
                # this annotation
                except KeyError:
                    continue

                # Get bbox and convert from x1,y1,w,h to x1,y1,x2,y2
                bbox = anno["bbox"]
                bbox[2] += bbox[0]
                bbox[3] += bbox[1]

                data_dict[img_id]["ann"]["bboxes"].append(bbox)
                data_dict[img_id]["ann"]["labels"].append(cls)


        # Convert data_dict to a list and all lists (bboxes and labels) to numpy arrays
        print("Converting and filtering...")
        for img_id in list(data_dict.keys()):
            val = data_dict[img_id]

            # Ignore the image if there is a strictly ignored class (-2)
            if -2 in val["ann"]["labels"]:
                strictly_ignored_counter += 1
                continue

            # Ignore images with no annotation (mmdetection can't handle them)
            if val["ann"]["labels"] == []:
                no_anno_counter += 1
                continue

            # Ignore images that have no vehicles
            if not coco_classes_map["car"] in val["ann"]["labels"] \
                    and not coco_classes_map["bus"] in val["ann"]["labels"] \
                    and not coco_classes_map["truck"] in val["ann"]["labels"]:
                no_vehicle_counter += 1
                continue

            # Update the filename to a relative path
            val["filename"] = os.path.join(imgs_dirnames[key], val["filename"])

            # Convert lists of bboxes and labels to arrays
            # Should work if done the same way as labels, but to be sure..:
            val["ann"]["bboxes"] = np.array(
                [np.array(l) for l in val["ann"]["bboxes"]], 
                dtype=np.int16)
            val["ann"]["labels"] = np.array(val["ann"]["labels"], dtype=np.int16)

            data_list.append(val)

    print(f"Saving {len(data_list)} annotated images")
    print(f"{strictly_ignored_counter} images discarded")
    print(f"{no_anno_counter} ignored since they contain no annotations")
    print(f"{no_vehicle_counter} ignored since they contain no vehicles")

    # Write the list to a file
    with open(gt_pickle_path, 'wb') as f:
        pickle.dump(data_list, f, protocol=common.pickle_file_protocol)

    print(f"Saved to {gt_pickle_path}")

if __name__ == "__main__":
    process_coco()