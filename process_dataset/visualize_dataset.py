import sys
import cv2
import pickle
import os
import shutil

import process_dataset.common as common


def visualize_dataset(dataset_name):

    print(f"Visualizing dataset {dataset_name}")

    if dataset_name not in common.datasets.keys():
        print(f"Dataset with name {dataset_name} not found in common.datasets")
        return
    dst_dirpath = os.path.join(common.datasets_path, "visualized_" + dataset_name)
    if os.path.exists(dst_dirpath):
        shutil.rmtree(dst_dirpath)
    os.mkdir(dst_dirpath)
        
    dataset_path = os.path.join(common.datasets_path, common.datasets[dataset_name]["path"])

    dataset_gt_filepath = os.path.join(dataset_path, common.gt_pickle_filename)

    print("Data loaded")

    with open(dataset_gt_filepath, "rb") as f:
        data = pickle.loads(f.read())
        for frame in data:
            old_filepath = os.path.join(dataset_path, frame["filename"])
            filename = os.path.basename(old_filepath)
            new_filepath = os.path.join(dst_dirpath, filename)
            bboxes = frame["ann"]["bboxes"]
            labels = frame["ann"]["labels"]
            img = cv2.imread(old_filepath)
            for i in range(len(bboxes)):
                if labels[i] == common.classes.index("passenger_car"):
                    color = (0, 255, 0)
                else:
                    color = (0, 0, 255)
                bbox = bboxes[i]
                label = labels[i]
                text = f"{common.classes_dict[label]} (bbox {i})"
                img = cv2.rectangle(img, bbox[:2], bbox[2:], color, 1)
                img = cv2.putText(img, text, (bbox[0], bbox[3]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
            cv2.imwrite(new_filepath, img)

    print("All done")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Please provide a dataset name as an argument")
        exit(1)
    visualize_dataset(sys.argv[1])