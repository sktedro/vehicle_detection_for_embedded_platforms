import os
import sys
import pickle
import json
from tqdm import tqdm

# The script should be importable but also executable from the terminal...
if __name__ == '__main__':
    import common
else:
    from . import common


def middleToCoco(input_filepath, output_filepath):

    print(f"Loading from {input_filepath}")
    with open(input_filepath, "rb") as input_file:
        data = pickle.load(input_file)

    # Make source a list of images (and their annotations)
    source = None
    if data == []:
        source = data
    else:
        for key in ["train", "val", "test"]:
            assert key in data.keys()
        source = data["train"] + data["val"] + data["test"]

    images = []
    annotations = []
    img_id_counter = 0
    ann_id_counter = 0
    print("Converting data")
    for img in tqdm(source):
        images.append({
            "id": img_id_counter,
            "width": img["width"],
            "height": img["height"],
            "file_name": img["filename"],
            "dataset_name": img["dataset_name"],
            # "date_captured": # Hopefully not required
        })

        for i in range(len(img["ann"]["labels"])):
            bbox = [int(coord) for coord in img["ann"]["bboxes"][i]]
            bbox[2] -= bbox[0] # Convert x2 to width
            bbox[3] -= bbox[1] # Convert y2 to height
            label = int(img["ann"]["labels"][i])
            annotations.append({
                "id": ann_id_counter,
                "image_id": img_id_counter,
                "bbox": bbox,
                "category_id": label,
            })
            ann_id_counter += 1

        img_id_counter += 1


    # Categories
    categories = []
    for key in list(common.classes_dict.keys()):
        categories.append({
            "name": common.classes_dict[key],
            "id": key,
            # "supercategory":
        })

    print(f"Writing to {output_filepath}")
    with open(output_filepath, "w") as output_file:
        output = json.dumps({
            "categories": categories,
            "images": images,
            "annotations": annotations,
        }, indent=2)
        output_file.write(output)

    
if __name__ == "__main__":
    if len(sys.argv) == 3:
        middleToCoco(sys.argv[1], sys.argv[2])
    else:
        middleToCoco(common.dataset_pickle_filepath, common.dataset_coco_filepath)