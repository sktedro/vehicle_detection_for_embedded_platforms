"""
Processes ground truth data of the NDISPark dataset (in COCO format):
- Fixes image IDs - they are only unique in their subsets, so they are
updated to be unique per dataset
- Fix annotations (since the dataset only uses one class) per class_fix var
- Converts bboxes to integers
- Updates filepaths to be relative to the dataset dirpath, not subset
dirpath
- And of course, maps the dataset's class to our class...
"""
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
# Also contains trailers, trucks and transporters. These are mapped manually 
# using class_fix var below
ndis_classes_map = {
    3: common.classes_ids["car"], # passenger car
}

# Dictionary containing image filenames as keys, and separate fixes as values.
# Fixes are in an array and one fix can be [annotation_id, new_class] or
# [[annotation_id_1, annotation_id_2], new_class]
# When processing, classes of objects are fixed using this.
class_fix = {
    '60_1537560142.jpg': [[372, 'transporter']],
    '60_1537641085.jpg': [[[1793, 1802, 1767], 'transporter']],
    '60_1537642922.jpg': [[[1907, 1924, 1927, 1943], 'transporter']],
    '60_1537644714.jpg': [[[2361, 2364, 2383], 'transporter']],
    '60_1537646501.jpg': [[[277, 278, 293], 'transporter']],
    '60_1537648298.jpg': [[182, 'transporter']],
    '60_1537700458.jpg': [[1353, 'transporter']],
    '60_1537702242.jpg': [[91, 'transporter']],
    '60_1537704079.jpg': [[1456, 'transporter']],
    '60_1537711240.jpg': [[497, 'trailer']],
    '60_1537713062.jpg': [[1631, 'trailer']],
    '60_1537714829.jpg': [[2234, 'trailer']],
    '60_1537716629.jpg': [[1874, 'trailer']],
    '60_1537718423.jpg': [[1293, 'trailer']],
    '62_1537439604.jpg': [[2003, 'transporter']],
    '62_1537610613.jpg': [[1887, 'transporter']],
    '62_1537639330.jpg': [[[936, 940], 'transporter']],
    '62_1537641142.jpg': [[[498, 499, 489, 468], 'transporter']],
    '62_1537642957.jpg': [[[2292, 2279, 2288, 2293, 2267], 'transporter']],
    '62_1537644729.jpg': [[[745, 764, 768, 754, 761], 'transporter']],
    '62_1537689729.jpg': [[1635, 'transporter']],
    '62_1537705928.jpg': [[579, 'transporter']],
    '62_1537709554.jpg': [[2036, 'trailer']],
    '62_1537713107.jpg': [[373, 'trailer']],
    '62_1537716658.jpg': [[1654, 'trailer']],
    '62_1537718436.jpg': [[1482, 'trailer']],
    '62_1537938236.jpg': [[626, 'transporter']],
    '62_1537956182.jpg': [[1127, 'transporter']],
    '62_1537959778.jpg': [[[2299, 2324], 'transporter']],
    '62_1537963395.jpg': [[2327, 'transporter']],
    '64_1531764036.jpg': [[207, 'transporter']],
    '64_1537009217.jpg': [[326, 'transporter']],
    '64_1537016417.jpg': [[311, 'transporter']],
    '64_1537032622.jpg': [[1959, 'transporter']],
    '64_1537097416.jpg': [[84, 'transporter']],
    '64_1537099216.jpg': [[2117, 'transporter']],
    '69_1531596734.jpg': [[625, 'transporter']],
    '69_1537347613.jpg': [[363, 'transporter']],
    '69_1537349415.jpg': [[2482, 'transporter']],
    '69_1537351214.jpg': [[[823, 824], 'transporter']],
    '69_1537353014.jpg': [[[898, 897], 'transporter']],
    '69_1537354811.jpg': [[[146, 145], 'transporter']],
    '69_1537358412.jpg': [[[705, 706], 'transporter']],
    '69_1537362013.jpg': [[171, 'transporter']],
    '73_1537095614.jpg': [[1024, 'transporter']],
    '73_1537108212.jpg': [[[441, 452], 'transporter']],
    '73_1537207256.jpg': [[2503, 'transporter']],
    '73_1537216265.jpg': [[381, 'transporter']],
    '73_1537239784.jpg': [[1807, 'transporter']],
    '73_1537252228.jpg': [[952, 'transporter']],
    '73_1537263016.jpg': [[[288, 290], 'transporter']],
    '73_1537286417.jpg': [[254, 'transporter']],
    '73_1537293677.jpg': [[1066, 'transporter']],
    '73_1537327869.jpg': [[2570, 'transporter']],
    '73_1537336816.jpg': [[946, 'transporter']],
    '73_1537347619.jpg': [[[1077, 1088], 'transporter']],
    '73_1537353020.jpg': [[2555, 'transporter']],
    '73_1537354816.jpg': [[[699, 658], 'transporter']],
    '73_1537414287.jpg': [[536, 'transporter']],
    '73_1537432229.jpg': [[1557, 'transporter']],
    '78_1531611010.jpg': [[1404, 'transporter']],
    '78_1531722605.jpg': [[312, 'transporter'], [307, 'trailer']],
    '78_1531809005.jpg': [[[316, 317], 'truck']],
    '78_1531814405.jpg': [[853, 'transporter']],
    '78_1531816205.jpg': [[2098, 'transporter']],
    '83_1531481408.jpg': [[[660, 666], 'transporter']],
    '83_1531486807.jpg': [[2560, 'transporter']],
    '83_1531497607.jpg': [[1294, 'transporter']],
    '83_1531510212.jpg': [[367, 'transporter']],
    '83_1531740623.jpg': [[[646, 641], 'transporter']],
    '83_1531751412.jpg': [[1719, 'transporter']],
    '83_1531755011.jpg': [[2239, 'transporter']],
    '83_1531764013.jpg': [[[516, 518], 'transporter']],
    '83_1531821617.jpg': [[135, 'transporter']]
    }

def process_ndis():
    # Initialize paths
    dataset_abs_dirpath = os.path.join(common.paths.datasets_dirpath, common.datasets["ndis"]["path"])
    gt_json_abs_filepaths = {
        "train": os.path.join(dataset_abs_dirpath, "train", "train_coco_annotations.json"),
        "val": os.path.join(dataset_abs_dirpath, "validation", "val_coco_annotations.json")
    }
    imgs_rel_dirpaths = {
        "train": os.path.join("train", "imgs"),
        "val": os.path.join("validation", "imgs")
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
                old_img_filepath = img["file_name"]
                new_img_filepath = os.path.join(imgs_rel_dirpaths[subset], img["file_name"])
                img["file_name"] = new_img_filepath

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
                if old_img_filepath in class_fix.keys():

                    for [anno_ids, new_cls] in class_fix[old_img_filepath]:

                        # If the anno_ids is not an array, make it an array...
                        if type(anno_ids) == int:
                            anno_ids = [anno_ids]

                        # For each annotation ID, fix the annotation
                        for anno_id in anno_ids:
                            for anno in annos:
                                if anno["id"] == anno_id:
                                    anno["category_id"] = common.classes_ids[new_cls]

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
