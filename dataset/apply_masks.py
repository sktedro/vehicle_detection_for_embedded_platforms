"""
Reads "combined" ground truth of datasets (all datasets if no arguments were
given, or datasets of which names were provided as arguments) and for every
image that has "mask" key or for every annotation with class "mask", applies the
mask to an image (all masked images are saved to a shared folder). The mask can
be a filepath (relative to the datasets dirpath) or an annotation. Finally,
filepaths ("file_name" keys) of images in all ground truth files (combined,
train, val, test) are updated

Parameters: dataset names as in common.datasets dict
"""
import os
import sys
import json
import shutil
from tqdm import tqdm

# The script should be importable but also executable from the terminal...
if __name__ == '__main__':
    import common
else:
    from . import common


# Choose to work with opencv-python or pillow. For masking with image file, both
# seem to be equally fast, but for masking with bboxes, PIL is about 3-4x faster
# Options: "pil" or "cv2"
backend = "pil"

if backend == "cv2":
    import cv2
    def mask_with_image(old_img_abs_filepath, new_img_abs_filepath, mask_abs_filepath):
        frame = cv2.imread(old_img_abs_filepath)
        mask = cv2.imread(mask_abs_filepath, cv2.COLOR_BGR2GRAY)
        frame = cv2.bitwise_and(frame, frame, mask=mask) # Apply mask
        cv2.imwrite(new_img_abs_filepath, frame)


    def mask_with_bboxes(old_img_abs_filepath, new_img_abs_filepath, bboxes):
        frame = cv2.imread(old_img_abs_filepath)
        for bbox in bboxes:
            frame = cv2.rectangle(frame, bbox[0:2], bbox[2:4], (0,0,0), -1)
        cv2.imwrite(new_img_abs_filepath, frame)

elif backend == "pil":
    from PIL import Image, ImageDraw
    def mask_with_image(old_img_abs_filepath, new_img_abs_filepath, mask_abs_filepath):
        old_img = Image.open(old_img_abs_filepath)
        mask = Image.open(mask_abs_filepath).convert('L')
        new_img = Image.new('RGB', old_img.size)
        new_img.paste(old_img, (0, 0), mask=mask)
        new_img.save(new_img_abs_filepath)


    def mask_with_bboxes(old_img_abs_filepath, new_img_abs_filepath, bboxes):
        frame = Image.open(old_img_abs_filepath)
        draw = ImageDraw.Draw(frame)
        for bbox in bboxes:
            draw.rectangle(bbox, outline=(0,0,0), fill=(0,0,0))
        frame.save(new_img_abs_filepath)


def apply_masks(dataset_name):

    print(f"Applying masks to dataset '{dataset_name}'")

    # Load the combined ground truth
    gt_unmasked_filepath = os.path.join(common.paths.datasets_dirpath,
                               common.datasets[dataset_name]["path"],
                               common.gt_unmasked_filenames["combined"])
    with open(gt_unmasked_filepath) as f:
        gt = json.load(f)

    # Create a directory for masked images (delete it first if exists)
    masked_imgs_abs_dirpath = os.path.join(common.paths.datasets_dirpath,
                                             common.datasets[dataset_name]["path"],
                                             common.masked_imgs_rel_dirpath)
    if os.path.exists(masked_imgs_abs_dirpath):
        shutil.rmtree(masked_imgs_abs_dirpath)
    os.mkdir(masked_imgs_abs_dirpath)

    print("Reading annotations of mask category")
    masks = {} # key = img_id, val = list of mask annotations
    for anno in tqdm(gt["annotations"]):
        if anno["category_id"] == common.classes_ids["mask"]:
            img_id = anno["image_id"]
            if img_id not in masks:
                masks[img_id] = [anno["bbox"]]
            else:
                masks[img_id].append(anno["bbox"])

    print("Applying masks to images")
    filenames_map = {}
    for img in tqdm(gt["images"]):
        if "mask" not in img and img["id"] not in masks:
            continue

        # Old relative image path
        old_img_rel_filepath = img["file_name"] # Relative to dataset
        old_img_rel_dirpath = os.path.dirname(old_img_rel_filepath)
        filename, ext = os.path.basename(old_img_rel_filepath).split(".")

        # New relative image path
        new_img_filename = str(img["id"]).zfill(6) + "_" + filename + "." + ext
        new_img_rel_filepath = os.path.join(common.masked_imgs_rel_dirpath,
                                            old_img_rel_dirpath,
                                            new_img_filename)

        # Old absolute image path
        old_img_abs_filepath = os.path.join(common.paths.datasets_dirpath,
                                        common.datasets[dataset_name]["path"],
                                        old_img_rel_filepath)
        # New absolute image path
        new_img_abs_filepath = os.path.join(common.paths.datasets_dirpath,
                                        common.datasets[dataset_name]["path"],
                                        new_img_rel_filepath)

        if not os.path.exists(os.path.dirname(new_img_abs_filepath)):
            os.makedirs(os.path.dirname(new_img_abs_filepath))

        if "mask" in img:
            mask_abs_filepath = os.path.join(common.paths.datasets_dirpath,
                                         common.datasets["aau"]["path"],
                                         img["mask"])
            mask_with_image(old_img_abs_filepath, new_img_abs_filepath, mask_abs_filepath)

        if img["id"] in masks:
            bboxes = masks[img["id"]]
            for bbox in bboxes: # x,y,w,h -> x1,y1,x2,y2
                bbox[2] += bbox[0]
                bbox[3] += bbox[1]
            mask_with_bboxes(old_img_abs_filepath, new_img_abs_filepath, bboxes)

        filenames_map[old_img_rel_filepath] = new_img_rel_filepath

        # Note: ground truth data will be updated later (when saving)

    print("Saving masked ground truth files")
    for subset in tqdm(["combined", "train", "val", "test"]):
        gt_unmasked_filepath = os.path.join(common.paths.datasets_dirpath,
                                            common.datasets[dataset_name]["path"],
                                            common.gt_unmasked_filenames[subset])
        gt_filepath = os.path.join(common.paths.datasets_dirpath,
                                   common.datasets[dataset_name]["path"],
                                   common.gt_filenames[subset])

        with open(gt_unmasked_filepath) as f:
            gt = json.load(f)

        for img in gt["images"]:
            if img["file_name"] in filenames_map:
                img["file_name"] = filenames_map[img["file_name"]]
                if "mask" in img:
                    del img["mask"]

        # Guess removing an item from a huge list takes a while...
        # for i in tqdm(range(len(gt["annotations"]))):
        #     if gt["annotations"][i]["category_id"] == common.classes_ids["mask"]:
        #         del gt["annotations"][i]
        #         i -= 1

        # So this approach is faster
        annos = []
        for anno in gt["annotations"]:
            if anno["category_id"] != common.classes_ids["mask"]:
                annos.append(anno)
        gt["annotations"] = annos

        with open(gt_filepath, "w") as f:
            json.dump(gt, f)

    # Remove the masked imgs dir if empty
    try:
        os.rmdir(masked_imgs_abs_dirpath)
    except OSError:
        pass

    print(f"Dataset '{dataset_name}': {len(filenames_map)} images masked to {masked_imgs_abs_dirpath}")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        dataset_names = sys.argv[1:]
    else:
        dataset_names = list(common.datasets.keys())
    print(f"Applying masks to datasets: {dataset_names}")
    for dataset_name in dataset_names:
        assert dataset_name in common.datasets, f"Dataset '{dataset_name}' not found"

        # import cProfile
        # import pstats
        # pr = cProfile.Profile()
        # pr.enable()
        # apply_masks(dataset_name)
        # pr.disable()
        # ps = pstats.Stats(pr)
        # ps.sort_stats('cumtime')
        # ps.print_stats(10)

        apply_masks(dataset_name)