import os
import json
import numpy as np
import pickle
import gc
import shutil

import common

mtid_classes_map = {
    2: common.classes.index("bicycle"),
    3: common.classes.index("passenger_car"),
    6: common.classes.index("bus"),
    8: common.classes.index("truck"),
}

def check_ignore(subset, img_id, img_rel_filepath=""):
    # Check if we're skipping this image because of the frame step
    if img_id % common.datasets["mtid"]["frame_step"] != 0:
        return True

    # Check if the image should be ignored due to it not being annotated
    if img_rel_filepath == "":
        return False

    for ignore_range in common.datasets["mtid"]["ignored_images_ranges"][subset]:
        for ignore_nr in range(ignore_range[0], ignore_range[1] + 1):
            if str(ignore_nr).zfill(7) + ".jpg" in img_rel_filepath:
                return True

    return False

def process_mtid():
    """Converts ground truth data of the MTID dataset from JSON (COCO format) to
    mmdetection's middle format in a pickle file

    MTID dataset (COCO) format (only showing fields of interes):
    ```json
    {
        "images": [
            {
                "id": 0,
                "width": 640,
                "height": 1024,
                "file_name": "Infrastructure/0/seq3-infra_0000001.jpg"
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
                "id": 2,
                "name": "bicycle"
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
    dataset_path = os.path.join(common.datasets_path, common.datasets["mtid"]["path"])
    gt_json_path = {
        "drone": os.path.join(dataset_path, "drone-mscoco.json"),
        "infra": os.path.join(dataset_path, "infrastructure-mscoco.json")
    }
    imgs_path = {
        "drone": os.path.join(dataset_path, "Drone/frames_masked"),
        "infra": os.path.join(dataset_path, "Infrastructure/frames_masked")
    }
    assert os.path.exists(imgs_path["drone"])
    assert os.path.exists(imgs_path["infra"])

    gt_pickle_path = os.path.join(dataset_path, common.gt_pickle_filename)

    # Create a directory for combined images (delete it first if exists)
    combined_imgs_rel_path = "imgs_combined"
    combined_imgs_path = os.path.join(dataset_path, combined_imgs_rel_path)
    if os.path.exists(combined_imgs_path):
        shutil.rmtree(combined_imgs_path)
    os.mkdir(combined_imgs_path)

    # Read the data while copying all images to combined/ directory, ignoring
    # images without annotations and while only copying only every Nth image
    # based on common.datasets["mtid"]["frame_step"]
    data_dict = {}
    for key in gt_json_path:

        try:
            max_id = max(list(data_dict.keys()))
        except:
            max_id = -1

        with open(gt_json_path[key]) as f:
            data = json.loads(f.read())

            # Fetch all annotations first and save them
            # Note: not ignoring images here to keep the code simpler (ignoring
            # in next for loop)
            for anno in data["annotations"]:

                # Check if we're skipping this image because of the frame step
                if anno["image_id"] % common.datasets["mtid"]["frame_step"] != 0:
                    continue

                # Note: ignoring images will be done in the next for loop,
                # through images

                new_img_id = anno["image_id"] // common.datasets["mtid"]["frame_step"] + max_id + 1

                # If the image is not yet in data_dict, initialize it
                if new_img_id not in data_dict.keys():
                    data_dict[new_img_id] = {
                        "ann": {
                            "bboxes": [],
                            "labels": []
                        }
                    }

                # Append annotation (class (label) and bbox)
                data_dict[new_img_id]["ann"]["labels"].append(mtid_classes_map[anno["category_id"]])

                # Convert bbox from [x1 y1 w h] to [x1 y1 x2 y2]
                bbox = anno["bbox"]
                bbox[2] = bbox[0] + bbox[2]
                bbox[3] = bbox[1] + bbox[3]
                data_dict[new_img_id]["ann"]["bboxes"].append(bbox)

            # Append info about the images while copying the images to combined/
            # Image: id, width, height, file_name
            for image in data["images"]:

                # Check if we're skipping this image because of the frame step
                if image["id"] % common.datasets["mtid"]["frame_step"] != 0:
                    continue

                new_img_id = image["id"] // common.datasets["mtid"]["frame_step"] + max_id + 1

                # Check if the image is annotated (look at
                # common.datasets["mtid"]["ignored_images_ranges"][subset]). If
                # not, delete the id from the data_dict since we didn't check if
                # it should be ignored when processing annotations (for code
                # simplicity)
                ignore = False
                for ignore_range in common.datasets["mtid"]["ignored_images_ranges"][key]:
                    for ignore_nr in range(ignore_range[0], ignore_range[1] + 1):
                        if str(ignore_nr).zfill(7) + ".jpg" in image["file_name"]:
                            ignore = True
                            if new_img_id in data_dict.keys():
                                del data_dict[new_img_id]
                            break
                    if ignore:
                        break
                if ignore:
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
                old_filename = os.path.basename(image["file_name"])
                old_masked_img_filepath = os.path.join(imgs_path[key], old_filename)
                new_img_filepath = os.path.join(dataset_path, new_img_rel_filepath)
                shutil.copy(old_masked_img_filepath, new_img_filepath)

                # print(f"{old_filename} -> {new_img_rel_filepath}")

    # We can free data from memory
    del data
    gc.collect()

    print(f"Data loaded and combined to {combined_imgs_path}")


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
    process_mtid()