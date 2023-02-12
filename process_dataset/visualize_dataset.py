import sys
import cv2
import json
import os
import shutil
from tqdm import tqdm

# The script should be importable but also executable from the terminal...
if __name__ == '__main__':
    import common
else:
    from . import common


def visualize_dataset(dataset_name):

    print(f"Visualizing dataset {dataset_name}")

    if dataset_name not in common.datasets.keys():
        print(f"Dataset with name {dataset_name} not found in common.datasets")
        return
    dst_dirpath = os.path.join(common.datasets_dirpath, "visualized_" + dataset_name)
    if os.path.exists(dst_dirpath):
        shutil.rmtree(dst_dirpath)
    os.mkdir(dst_dirpath)
        
    dataset_abs_dirpath = os.path.join(common.datasets_dirpath, common.datasets[dataset_name]["path"])

    dataset_gt_filepath = os.path.join(dataset_abs_dirpath, common.gt_filename)

    with open(dataset_gt_filepath, "r") as f:
        data = json.loads(f.read())

        for img in tqdm(data["images"]):

            img_id = img["id"]

            old_abs_filepath = os.path.join(common.datasets_dirpath, img["file_name"])
            filename = os.path.basename(old_abs_filepath)
            new_abs_filepath = os.path.join(dst_dirpath, filename)

            assert os.path.exists(old_abs_filepath)
            img = cv2.imread(old_abs_filepath)
            for anno in data["annotations"]:
                if anno["image_id"] == img_id:
                    bbox = anno["bbox"]
                    bbox[2] += bbox[0]
                    bbox[3] += + bbox[1]
                    cls = anno["category_id"]
                    if cls == common.classes_ids["car"]:
                        color = (0, 255, 0)
                    else:
                        color = (0, 0, 255)
                    text = f"{common.classes_names[cls]} (bbox {anno['id']})"
                    img = cv2.rectangle(img, bbox[:2], bbox[2:], color, 1)
                    img = cv2.putText(img, text, (bbox[0], bbox[3]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

            cv2.imwrite(new_abs_filepath, img)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Please provide a dataset name as an argument")
        exit(1)
    visualize_dataset(sys.argv[1])