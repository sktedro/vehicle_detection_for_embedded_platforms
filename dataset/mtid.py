import os
import json
import shutil
from tqdm import tqdm

# The script should be importable but also executable from the terminal...
if __name__ == '__main__':
    import common
else:
    from . import common

mtid_classes_map = {
    2: common.classes_ids["bicycle"],
    3: common.classes_ids["car"],
    6: common.classes_ids["bus"],
    8: common.classes_ids["truck"]
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
    """Processes ground truth data of the MTID dataset from JSON (COCO format):
    - Combines both subsets (drone, infrastructure) - both subsets have separate
    image IDs, so these IDs are changed to be unique in both subsets
    - Ignores frames as selected in common.py
    """

    # Initialize paths
    dataset_abs_dirpath = os.path.join(common.paths.datasets_dirpath, common.datasets["mtid"]["path"])
    gt_json_abs_filepaths = {
        "drone": os.path.join(dataset_abs_dirpath, "drone-mscoco.json"),
        "infra": os.path.join(dataset_abs_dirpath, "infrastructure-mscoco.json")
    }
    imgs_abs_dirpaths = {
        "drone": os.path.join(dataset_abs_dirpath, "Drone/frames_masked"),
        "infra": os.path.join(dataset_abs_dirpath, "Infrastructure/frames_masked")
    }
    assert os.path.exists(gt_json_abs_filepaths["drone"])
    assert os.path.exists(gt_json_abs_filepaths["infra"])
    assert os.path.exists(imgs_abs_dirpaths["drone"])
    assert os.path.exists(imgs_abs_dirpaths["infra"])

    # Create a directory for combined images (delete it first if exists)
    combined_imgs_rel_dirpath = "imgs_combined"
    combined_imgs_abs_dirpath = os.path.join(dataset_abs_dirpath, combined_imgs_rel_dirpath)
    if os.path.exists(combined_imgs_abs_dirpath):
        shutil.rmtree(combined_imgs_abs_dirpath)
    os.mkdir(combined_imgs_abs_dirpath)

    # Read the data while copying all images to combined/ directory, ignoring
    # images without annotations and while only copying only every Nth image
    # based on common.datasets["mtid"]["frame_step"]
    data = {
        "images": [],
        "annotations": []
    }

    img_id_counter = 0
    img_id_map = {} # Mapping old (subset dependant) IDs to new ones

    anno_id_counter = 0

    for subset in gt_json_abs_filepaths:

        with open(gt_json_abs_filepaths[subset]) as f:
            data_input = json.loads(f.read())

            # Get image info while copying the images to combined/ and ignoring
            # ones stated in common.py
            print(f"Processing images in {subset}")
            for image in tqdm(data_input["images"]):

                # Check if the image is annotated (look at
                # common.datasets["mtid"]["ignored_images_ranges"][subset]). If
                # not, ignore it
                ignore = False
                for ignore_range in common.datasets["mtid"]["ignored_images_ranges"][subset]:
                    for ignore_number in range(ignore_range[0], ignore_range[1] + 1):
                        if str(ignore_number).zfill(7) + ".jpg" in image["file_name"]:
                            ignore = True
                            break
                    if ignore:
                        break
                if ignore:
                    continue

                old_img_id = image["id"]
                new_img_id = img_id_counter
                img_id_map[old_img_id] = new_img_id
                img_id_counter += 1

                new_img_rel_filepath = os.path.join(common.datasets["mtid"]["path"], combined_imgs_rel_dirpath, str(new_img_id).zfill(9) + ".jpg")

                data["images"].append({
                    "id": new_img_id,
                    "height": image["height"],
                    "width": image["width"],
                    "file_name": new_img_rel_filepath
                })

                # Copy image to combined/
                old_filename = os.path.basename(image["file_name"])
                old_img_abs_filepath = os.path.join(imgs_abs_dirpaths[subset], old_filename)
                new_img_abs_filepath = os.path.join(common.paths.datasets_dirpath, new_img_rel_filepath)
                shutil.copy(old_img_abs_filepath, new_img_abs_filepath)

            print(f"Reading annotations in {subset} subset")
            for anno in tqdm(data_input["annotations"]):

                old_img_id = anno["image_id"]

                # Ignore annotations to images that were not processed (should
                # be ignored)
                if old_img_id not in list(img_id_map.keys()):
                    continue

                new_img_id = img_id_map[old_img_id]

                data["annotations"].append({
                    "id": anno_id_counter,
                    "image_id": new_img_id,
                    "category_id": mtid_classes_map[anno["category_id"]],
                    "bbox": anno["bbox"]
                })

                anno_id_counter += 1

    common.save_processed("mtid", data)


if __name__ == "__main__":
    process_mtid()
