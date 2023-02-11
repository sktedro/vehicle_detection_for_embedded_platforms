import os
import csv
import cv2
from tqdm import tqdm

# The script should be importable but also executable from the terminal...
if __name__ == '__main__':
    import common
else:
    from . import common


mio_tcd_classes_map = {
    "pedestrian":            -1, # Ignore
    "bicycle":               common.classes_ids["bicycle"],
    "motorcycle":            common.classes_ids["motorcycle"],
    "car":                   common.classes_ids["passenger_car"],
    "non-motorized_vehicle": common.classes_ids["trailer"],
    "pickup_truck":          common.classes_ids["transporter"],
    "work_van":              common.classes_ids["transporter"],
    "bus":                   common.classes_ids["bus"],
    "articulated_truck":     common.classes_ids["truck"],
    "single_unit_truck":     common.classes_ids["truck"],
    "motorized_vehicle":     common.classes_ids["unknown"] # Vehicles that cannot be classified (mostly bad video quality)
}


def process_mio_tcd():
    """Converts ground truth data of the MIO-TCD dataset's train subset from CSV
    to COCO format

    This can take several minutes

    Part of code taken from view_bounding_boxes.py script provided by creators of MIO-TCD

    MIO-TCD format:
    `00000000, pickup_truck, 213, 34, 255, 50` as 
    `a, label, x1, y1, x2, y2`

    or, if the row has 7 values:
    `00000000, pickup_truck, 0.9, 213, 34, 255, 50` as 
    `a, label, score, x1, y1, x2, y2`
    """

    # Initialize paths
    # dataset_path = common.datasets["mio-tcd"]["path"]
    dataset_abs_dirpath = os.path.join(common.datasets_dirpath, common.datasets["mio-tcd"]["path"])
    gt_csv_abs_filepath = os.path.join(dataset_abs_dirpath, "gt_train.csv")
    imgs_abs_dirpath = os.path.join(dataset_abs_dirpath, "train")

    lines_total = len(["" for _ in open(gt_csv_abs_filepath)]) # Because len() doesn't work

    data = {
        "images": [],
        "annotations": []
    }
    imgs_processed = []

    anno_id_counter = 0

    print(f"Processing {gt_csv_abs_filepath}")
    with open(gt_csv_abs_filepath) as csv_f:
        reader = csv.reader(csv_f, delimiter=',')

        for row in tqdm(reader, total=lines_total):
            
            img_id_str = row[0]
            img_filename = img_id_str + ".jpg"
            cls = mio_tcd_classes_map[row[1]]

            # Add image info to data var if not already there
            if img_filename not in imgs_processed:
                imgs_processed.append(img_filename)

                # Height and width of an image
                height, width, _ = cv2.imread(os.path.join(imgs_abs_dirpath, img_filename)).shape

                data["images"].append({
                    "id": int(img_id_str),
                    "file_name": os.path.join(common.datasets["mio-tcd"]["path"], "train", img_filename),
                    "width": width,
                    "height": height,
                })
            
            # If class is -1, ignore this annotation
            if cls != -1:
                # Bounding box
                if len(row) == 6:
                    bbox = [int(val) for val in row[2:]]
                elif len(row) == 7:
                    # Skip score field at 2nd index
                    bbox = [int(val) for val in row[3:]]
                else:
                    raise Exception("Encountered a row with length != 6 and != 7")

                # Convert from x1,y1,x2,y2 -> x,y,w,h
                bbox[2] = bbox[2] - bbox[0]
                bbox[3] = bbox[3] - bbox[1]
                
                data["annotations"].append({
                    "id": anno_id_counter,
                    "image_id": int(img_id_str),
                    "category_id": cls,
                    "bbox": bbox
                })

                anno_id_counter += 1

    common.save_processed("mio-tcd", data)

if __name__ == "__main__":
    process_mio_tcd()