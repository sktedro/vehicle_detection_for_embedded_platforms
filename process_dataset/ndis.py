import os
import json
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

# Dictionary containing image filenames as keys, and separate fixes as values.
# Fixes are in an array and one fix can be [bbox_number, new_class] or
# [[img_number1, img_number2], new_class]
# When processing, classes of bboxes for the filenames are fixed using this.
# Mind that img_numbers are not image IDs, although they should be! TODO update to img IDs
class_fix = {
    '60_1537560142.jpg': [[0, "transporter"]],
    '60_1537641085.jpg': [[[36, 45, 10], "transporter"]],
    '60_1537642922.jpg': [[[4, 21, 24, 40], "transporter"]],
    '60_1537644714.jpg': [[[9, 12, 31], "transporter"]],
    '60_1537646501.jpg': [[[13, 14, 29], "transporter"]],
    '60_1537648298.jpg': [[3, "transporter"]],
    '60_1537700458.jpg': [[53, "transporter"]],
    '60_1537702242.jpg': [[45, "transporter"]],
    '60_1537704079.jpg': [[46, "transporter"]],
    '60_1537711240.jpg': [[59, "trailer"]],
    '60_1537713062.jpg': [[71, "trailer"]],
    '60_1537714829.jpg': [[69, "trailer"]],
    '60_1537716629.jpg': [[65, "trailer"]],
    '60_1537718423.jpg': [[60, "trailer"]],
    '62_1537439604.jpg': [[2, "transporter"]],
    '62_1537610613.jpg': [[1, "transporter"]],
    '62_1537639330.jpg': [[[25, 29], "transporter"]],
    '62_1537641142.jpg': [[[33, 34, 24, 3], "transporter"]],
    '62_1537642957.jpg': [[[41, 28, 37, 42, 16], "transporter"]],
    '62_1537644729.jpg': [[[16, 35, 39, 25, 32], "transporter"]],
    '62_1537689729.jpg': [[0, "transporter"]],
    '62_1537705928.jpg': [[36, "transporter"]],
    '62_1537709554.jpg': [[1, "trailer"]],
    '62_1537713107.jpg': [[1, "trailer"]],
    '62_1537716658.jpg': [[1, "trailer"]],
    '62_1537718436.jpg': [[1, "trailer"]],
    '62_1537938236.jpg': [[0, "transporter"]],
    '62_1537956182.jpg': [[3, "transporter"]],
    '62_1537959778.jpg': [[[1, 26], "transporter"]],
    '62_1537963395.jpg': [[1, "transporter"]],
    '64_1531764036.jpg': [[0, "transporter"]],
    '64_1537009217.jpg': [[0, "transporter"]],
    '64_1537016417.jpg': [[1, "transporter"]],
    '64_1537032622.jpg': [[7, "transporter"]],
    '64_1537097416.jpg': [[15, "transporter"]],
    '64_1537099216.jpg': [[16, "transporter"]],
    '69_1531596734.jpg': [[11, "transporter"]],
    '69_1537347613.jpg': [[11, "transporter"]],
    '69_1537349415.jpg': [[15, "transporter"]],
    '69_1537351214.jpg': [[[4, 5], "transporter"]],
    '69_1537353014.jpg': [[[19, 18], "transporter"]],
    '69_1537354811.jpg': [[[5, 4], "transporter"]],
    '69_1537358412.jpg': [[[5, 6], "transporter"]],
    '69_1537362013.jpg': [[2, "transporter"]],
    '73_1537095614.jpg': [[8, "transporter"]],
    '73_1537108212.jpg': [[[3, 14], "transporter"]],
    '73_1537207256.jpg': [[1, "transporter"]],
    '73_1537216265.jpg': [[6, "transporter"]],
    '73_1537239784.jpg': [[4, "transporter"]],
    '73_1537252228.jpg': [[1, "transporter"]],
    '73_1537263016.jpg': [[[6, 8], "transporter"]],
    '73_1537286417.jpg': [[4, "transporter"]],
    '73_1537293677.jpg': [[6, "transporter"]],
    '73_1537327869.jpg': [[5, "transporter"]],
    '73_1537336816.jpg': [[2, "transporter"]],
    '73_1537347619.jpg': [[[6, 17], "transporter"]],
    '73_1537353020.jpg': [[42, "transporter"]],
    '73_1537354816.jpg': [[[42, 1], "transporter"]],
    '73_1537414287.jpg': [[6, "transporter"]],
    '73_1537432229.jpg': [[10, "transporter"]],
    '78_1531611010.jpg': [[4, "transporter"]],
    '78_1531722605.jpg': [[5, "transporter"], [0, "trailer"]],
    '78_1531809005.jpg': [[[2, 3], "truck"]],
    '78_1531814405.jpg': [[3, "transporter"]],
    '78_1531816205.jpg': [[7, "transporter"]],
    '83_1531481408.jpg': [[[0, 6], "transporter"]],
    '83_1531486807.jpg': [[4, "transporter"]],
    '83_1531497607.jpg': [[0, "transporter"]],
    '83_1531510212.jpg': [[6, "transporter"]],
    '83_1531740623.jpg': [[[6, 1], "transporter"]],
    '83_1531751412.jpg': [[4, "transporter"]],
    '83_1531755011.jpg': [[2, "transporter"]],
    '83_1531764013.jpg': [[[1, 3], "transporter"]],
    '83_1531821617.jpg': [[2, "transporter"]],

}


def process_ndis():
    """Processes ground truth data of the NDISPark dataset (in COCO format):
    - Fixes image IDs - they are only unique in their subsets, so they are
    updated to be unique per dataset
    - Fix annotations (since the dataset only uses one class) per class_fix var
    - Converts bboxes to integers
    - Updates filepaths to be relative to the dataset dirpath, not subset
    dirpath
    - And of course, maps the dataset's class to our class...
    """

    # Initialize paths
    dataset_abs_dirpath = os.path.join(common.datasets_dirpath, common.datasets["ndis"]["path"])
    gt_json_abs_filepaths = {
        "train": os.path.join(dataset_abs_dirpath, "train/train_coco_annotations.json"),
        "val": os.path.join(dataset_abs_dirpath, "validation/val_coco_annotations.json")
    }
    imgs_rel_dirpaths = {
        "train": os.path.join(common.datasets["ndis"]["path"], "train/imgs"),
        "val": os.path.join(common.datasets["ndis"]["path"], "validation/imgs")
    }

    data = {
        "images": [],
        "annotations": []
    }

    # Since we're combining the datasets, we need to assign new IDs
    img_id_counter = 0
    anno_id_counter = 0

    for subset in gt_json_abs_filepaths:
        
        with open(gt_json_abs_filepaths[subset]) as f:
            data_input = json.loads(f.read())

            # Save misc data
            data["info"] = data_input["info"]
            data["licenses"] = data_input["licenses"]

            # Process the subset by cycling through images for each image,
            # processing annotations.
            # This is inefficient, but that's not a problem for dataset as small
            # as this, and there's a reason it is done this way...
            print(f"Processing subset {subset}")
            for img in tqdm(data_input["images"]):

                # Update image ID to be unique across subsets
                old_img_id = img["id"]
                new_img_id = img_id_counter
                img["id"] = new_img_id

                # Update the filepath to be relative to the datasets dir, not just the filename
                old_img_file_name = img["file_name"]
                new_img_filename = os.path.join(imgs_rel_dirpaths[subset], img["file_name"])
                img["file_name"] = new_img_filename

                # Save the image to data var
                data["images"].append(img)

                # Get all annotations
                annos = []
                for anno in data_input["annotations"]:
                    if anno["image_id"] == old_img_id:
                        annos.append(anno)

                # Process annotations
                for anno in annos:

                    # Map classes of annotations
                    anno["category_id"] = ndis_classes_map[anno["category_id"]]

                    # Convert bbox to ints
                    for i in range(4):
                        anno["bbox"][i] = int(anno["bbox"][i])

                    # Update img ID
                    anno["image_id"] = new_img_id

                # Fix annotations as specified in class_fix var
                if old_img_file_name in class_fix.keys():

                    for [bbox_ids, new_cls] in class_fix[old_img_file_name]:

                        # If the bbox_ids is not an array, make it an array...
                        if type(bbox_ids) == int:
                            bbox_ids = [bbox_ids]

                        # For each bbox ID, fix the annotation
                        for bbox_id in bbox_ids:
                            annos[bbox_id]["category_id"] = common.classes.index(new_cls)

                # Update annotations' IDs to be unique across subsets
                for anno in annos:
                    anno["id"] = anno_id_counter
                    anno_id_counter += 1

                # Save the annotations to data var
                for anno in annos:
                    data["annotations"].append(anno)

                img_id_counter += 1
    
    common.save_processed("ndis", data)

if __name__ == "__main__":
    process_ndis()