"""
Converts ground truth data of the VisDrone DET dataset's train subset
from CSV to COCO format

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
"""
import os
from PIL import Image
import csv
from tqdm import tqdm

# The script should be importable but also executable from the terminal...
if __name__ == '__main__':
    import common
else:
    from . import common


visdrone_det_classes_map = {
    "0":  common.classes_ids["mask"], # "ignored region" are masks
    "1":  -1, # "pedestrian" - ignore
    "2":  -1, # "people" - ignore
    "3":  common.classes_ids["bicycle"],
    "4":  common.classes_ids["car"],
    "5":  common.classes_ids["transporter"],
    "6":  common.classes_ids["truck"],
    "7":  common.classes_ids["unknown"], # Tricycle
    "8":  common.classes_ids["unknown"], # Awning-tricycle
    "9":  common.classes_ids["bus"],
    "10": common.classes_ids["motorcycle"], # "motor" is a motorcycle
    "11": -1 # Ignore "others". I don't remember what the objects are, but probably non-vehicles TODO check
}


def process_visdrone_det():
    # Initialize paths
    dataset_abs_dirpath = os.path.join(common.paths.datasets_dirpath, common.datasets["visdrone_det"]["path"])
    gts_abs_dirpath = os.path.join(dataset_abs_dirpath, "annotations")
    imgs_abs_dirpath = os.path.join(dataset_abs_dirpath, "images")

    gts_files_list = os.listdir(gts_abs_dirpath)
    imgs_files_list = os.listdir(imgs_abs_dirpath)

    img_id_counter = 0
    anno_id_counter = 0

    data = {
        "images": [],
        "annotations": []
    }

    print("Loading and processing annotations")
    for gt_file in tqdm(gts_files_list): # One gt file per image

        img_filename = gt_file.split(".")[0] + ".jpg"

        # Check that image exists
        # assert img_filename in imgs_files_list
        if img_filename not in imgs_files_list:
            print(f"Could not find image \"{img_filename}\". Ignoring")
            continue

        # Height and width of an image
        width, height = Image.open(os.path.join(imgs_abs_dirpath, img_filename)).size

        data["images"].append({
            "id": img_id_counter,
            "file_name": os.path.join("images", img_filename),
            "width": width,
            "height": height,
        })

        gt_abs_filepath = os.path.join(gts_abs_dirpath, os.path.basename(gt_file))
        with open(gt_abs_filepath) as csv_f:
            reader = csv.reader(csv_f, delimiter=',')

            for row in reader:

                bbox_left, bbox_top, w, h, _, cls, _, _ = row

                # Vehicle class (while mapping from MIO-TCD to ours representation)
                cls = visdrone_det_classes_map[cls]

                # If class is -1, ignore this annotation
                if cls != -1: 

                    data["annotations"].append({
                        "id": anno_id_counter,
                        "image_id": img_id_counter,
                        "category_id": cls,
                        "bbox": [int(bbox_left), int(bbox_top), int(w), int(h)]
                    })

                    anno_id_counter += 1

        img_id_counter += 1

    common.save_processed("visdrone_det", data)

if __name__ == "__main__":
    process_visdrone_det()
