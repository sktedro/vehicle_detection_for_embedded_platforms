import argparse
import json
import os
import re
import matplotlib.pyplot as plt
import numpy as np

from dataset import common as dataset_common


model_name_regex = r".*working_dir_(.*)_(\d+)x(\d+).*"
model_names = {
    "yolov8_f": "YOLOv8-femto",
    "yolov8_p": "YOLOv8-pico",
    "yolov8_n": "YOLOv8-nano",
    "yolov8_s": "YOLOv8-small",
    "yolov8_m": "YOLOv8-medium",
    "yolov8_l_mobilenet_v2": "YOLOv8 MobileNetV2"
}

def main(work_dirs):
    plt.rcParams.update({"font.size": 10})
    plt.rcParams.update({"font.family": "serif"})
    plt.rcParams.update({"font.serif": "cmr10"})
    plt.rcParams.update({"text.usetex": True})

    num_work_dirs = len(work_dirs)

    if len(work_dirs) == 9:
        num_rows = int(np.ceil(num_work_dirs / 3))
        fig, axes = plt.subplots(num_rows, 3, figsize=(6.5, 2.25 * num_rows), sharex=True, sharey=True)
    else:
        num_rows = int(np.ceil(num_work_dirs / 2))
        fig, axes = plt.subplots(num_rows, 2, figsize=(5, 2.5 * num_rows), sharex=True, sharey=True)

    axes = axes.flatten()

    for idx, work_dir in enumerate(work_dirs):
        ax = axes[idx]

        with open(os.path.join(work_dir, "coco_eval.json")) as f:
            precision_and_recall = json.load(f)

        precisions = np.array(precision_and_recall['precision'])
        recalls = np.array(precision_and_recall['recall'])
        scores = precision_and_recall['scores']
        iou_thresholds = precision_and_recall['iouThrs']
        record_thresholds = precision_and_recall['recThrs']
        category_ids = precision_and_recall['catIds']
        max_dets = precision_and_recall['maxDets']
        area_ranges = precision_and_recall['areaRng']
        area_ranges_labels = precision_and_recall['areaRngLbl']

        num_classes = len(category_ids) - 1 # Ignore mask class
        area_ranges_index = 0  # "all"
        max_dets_index = 2  # 100

        precision = precisions[:, :, :, area_ranges_index, max_dets_index]
        avg_precision = precision.mean(axis=0)

        recall_per_conf_threshold = np.linspace(0, 1, num=len(record_thresholds))
        recalls_expanded = np.tile(recall_per_conf_threshold[:, np.newaxis], (1, 9))

        for category_id, (category_precision, category_recall) in enumerate(zip(avg_precision.T, recalls_expanded.T)):
            cls_name = dataset_common.classes_names[category_id + 1]
            if cls_name == "mask":
                continue
            else:
                cls_name = cls_name.capitalize()
            if np.all(np.array(category_precision) == -1):
                category_precision = len(category_precision) * [0]
            ax.plot(category_recall, category_precision, label=cls_name)

        model_name, w, h = re.match(model_name_regex, work_dir).groups()
        ax.set_title(f"{model_names[model_name]} ${w}" + r"\times" + f"{h}$", fontsize=12)

        ax.grid(True, linestyle="--", alpha=0.3)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

    plt.tight_layout()
    fig.text(0.5, 0, 'Recall', ha='center')
    fig.text(0, 0.5, 'Precision', va='center', rotation='vertical')
    if num_rows == 1:
        leg = plt.legend(bbox_to_anchor=(1, 1), loc='upper left', borderaxespad=0., bbox_transform=fig.transFigure)
    elif num_rows == 2:
        leg = plt.legend(bbox_to_anchor=(1.05, 2.175), loc='upper left', borderaxespad=0.)
    else:
        leg = plt.legend(bbox_to_anchor=(1.05, 1.15 * num_rows), loc='upper left', borderaxespad=0.)
    for legobj in leg.legendHandles:
        legobj.set_linewidth(3.0)

    # xticks = [0, 0.2, 0.4, 0.6, 0.8, 1]
    # ax.set_xticks(xticks)
    # ax.set_xticklabels([str(t) for t in xticks])

    if len(work_dirs) == 4:
        fig.suptitle("Precision-Recall Curves of Major Models", fontsize=14, y=1.02)
    else:
        fig.suptitle("Additional Precision-Recall Curves", fontsize=16, y=1.02)

    # Save the plot to the first work_dir in the list
    if len(work_dirs) == 1:
        plt.savefig(os.path.join(work_dirs[0], "pr_curve.pdf"), bbox_inches='tight', transparent=True)
    else:
        plt.savefig("pr_curve.pdf", bbox_inches='tight', transparent=True)





if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("work_dirs", type=str, nargs="+",
                        help="working dirpaths. If multiple given, the PR curve will be saved to the local dir containing all models")
    args = parser.parse_args()

    # assert len(args.work_dirs) in [1, 2, 4, 6, 8]
    for work_dir in args.work_dirs:
        assert os.path.exists(work_dir)

    main(args.work_dirs)
