import os
import cv2
import json
import shutil
from tqdm import tqdm

# The script should be importable but also executable from the terminal...
if __name__ == '__main__':
    import common
else:
    from . import common

# Classes persent in the dataset: 1, 2, 3, 4, 6, 8, or in text format:
# person, bicycle, car, motorbike, bus, truck
aau_classes_map = {
    1: -1,                               # person
    2: common.classes_ids["bicycle"],    # bicycle
    4: common.classes_ids["motorcycle"], # motorbike
    3: common.classes_ids["car"],        # passenger car
    6: common.classes_ids["bus"],        # bus
    8: common.classes_ids["truck"]       # truck
}


def process_aau():
    """Processes the AAU RainSnow dataset, which is in COCO format, mapping
    classes and updating file paths. Saves ground truth in a COCO format. Also
    applies masks to images and combines them to a separate folder
    """

    # Initialize paths
    dataset_abs_dirpath = os.path.join(common.datasets_dirpath, common.datasets["aau"]["path"])
    gt_json_abs_filepath = os.path.join(dataset_abs_dirpath, "aauRainSnow-rgb.json")

    # For some reason, Egensevej-5 mask has the wrong filename - fix that:
    if not os.path.exists(os.path.join(dataset_abs_dirpath, "Egensevej", "Egensevej-5-mask.png")):
        dirpath = os.path.join(dataset_abs_dirpath, "Egensevej")
        src = os.path.join(dirpath, "Egensevej-5.png")
        dst = os.path.join(dirpath, "Egensevej-5-mask.png")
        shutil.copy(src, dst)

    # Create a directory for combined images (delete it first if exists)
    combined_imgs_rel_dirpath = "imgs_combined"
    combined_imgs_abs_dirpath = os.path.join(dataset_abs_dirpath, combined_imgs_rel_dirpath)
    if os.path.exists(combined_imgs_abs_dirpath):
        shutil.rmtree(combined_imgs_abs_dirpath)
    os.mkdir(combined_imgs_abs_dirpath)


    print(f"Loading data from {gt_json_abs_filepath}")
    with open(os.path.join(gt_json_abs_filepath)) as f:
        data = json.loads(f.read())

    print("Removing ignored images")
    for img in tqdm(data["images"].copy()):
        ignore = False
        for ignore_str in common.datasets["aau"]["ignored_folders"]:
            if ignore_str in img["file_name"]:
                data["images"].remove(img)
                ignore = True
                break
        if ignore:
            continue

    print(f"Applying masks, combining to {combined_imgs_abs_dirpath} and updating filenames")
    for img in tqdm(data["images"]):

        # Copy all images to a separate folder while applying detection masks (so
        # that the image only contains annotated vehicles) and update the paths
        old_img_rel_filepath = os.path.join(dataset_abs_dirpath, img["file_name"])
        new_img_rel_filepath = os.path.join(common.datasets["aau"]["path"], combined_imgs_rel_dirpath, str(img["id"]).zfill(9) + ".jpg")

        footage_dirpath = os.path.dirname(old_img_rel_filepath) # .../Egensevej-1
        location_dirpath = os.path.dirname(footage_dirpath) # .../Evensevej

        # Mask filename example: Evensevej-1-mask.jpg in the same directory as
        # Egensevej-1/ directory
        mask_filename = os.path.basename(footage_dirpath) + "-mask.png"
        mask_filepath = os.path.join(location_dirpath, mask_filename)

        frame = cv2.imread(old_img_rel_filepath)
        mask = cv2.imread(mask_filepath)
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        new_frame = cv2.bitwise_and(frame, frame, mask=mask) # Apply mask

        cv2.imwrite(os.path.join(common.datasets_dirpath, new_img_rel_filepath), new_frame)

        img["file_name"] = new_img_rel_filepath

    print("Removing ignored annotations")
    img_ids = []
    for img in data["images"]:
        img_ids.append(img["id"])
    for anno in tqdm(data["annotations"].copy()):
        if anno["image_id"] not in img_ids:
            data["annotations"].remove(anno)

    print("Removing annotations with ignored classes")
    for anno in tqdm(data["annotations"].copy()):
        if aau_classes_map[anno["category_id"]] == -1:
            data["annotations"].remove(anno)

    print("Removing invalid annotations")
    for anno in tqdm(data["annotations"].copy()):
        if anno["bbox"][2] < 0 or anno["bbox"][3] < 0:
            data["annotations"].remove(anno)

    print("Mapping classes")
    for anno in tqdm(data["annotations"]):
        anno["category_id"] = aau_classes_map[anno["category_id"]]

    # Convert data_dict to a list and all lists (bboxes and labels) to numpy arrays
    common.save_processed("aau", data)


if __name__ == "__main__":
    process_aau()