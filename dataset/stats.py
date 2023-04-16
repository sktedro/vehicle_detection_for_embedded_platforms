"""
Counts class instances and number of images in all datasets - combined, train,
val and test subset
"""
import json
import os

# The script should be importable but also executable from the terminal...
if __name__ == '__main__':
    import common
else:
    from . import common


def stats():

    print("Counting instances and images in processed datasets...", end="\r")

    legend = ["class_name", "class_id", "instances_total", "instances_train", "instances_val", "instances_test"]

    classes_counts = {
        "train": [0] * len(common.classes_ids),
        "val": [0] * len(common.classes_ids),
        "test": [0] * len(common.classes_ids),
    }
    images_counts = {
        "train": 0,
        "val": 0,
        "test": 0
    }

    for subset in ["train", "val", "test"]:
        gt_filepath = os.path.join(common.paths.datasets_dirpath, common.gt_combined_filenames[subset])
        with open(gt_filepath) as f:
            data = json.loads(f.read())
            images_counts[subset] = len(data["images"])
            for anno in data["annotations"]:
                classes_counts[subset][anno["category_id"] - 1] += 1


    print("Number of images:" + 50 * " ")
    print(80 * "=")

    table = [
        ["images_total", "images_train", "images_val", "images_test"],
        [images_counts["train"] + images_counts["val"] + images_counts["test"],
            images_counts["train"],
            images_counts["val"],
            images_counts["test"]]
    ]

    for row in table:
        print(str(row[0]).ljust(20)
              + str(row[1]).rjust(20)
              + str(row[2]).rjust(20)
              + str(row[3]).rjust(20)
        )

    print()


    print("Class instances in all images:")
    print(115 * "=")

    table = [legend]
    for class_id in list(common.classes_names.keys()):
        table.append([
            common.classes_names[class_id],
            class_id,
            classes_counts["train"][class_id - 1] + classes_counts["val"][class_id - 1] + classes_counts["test"][class_id - 1],
            classes_counts["train"][class_id - 1],
            classes_counts["val"][class_id - 1],
            classes_counts["test"][class_id - 1],
        ])
    table.append([
        "sum",
        "",
        sum(classes_counts["train"]) + sum(classes_counts["val"]) + sum(classes_counts["test"]),
        sum(classes_counts["train"]),
        sum(classes_counts["val"]),
        sum(classes_counts["test"]),
    ])

    for row in table:
        print(row[0].ljust(20)
              + str(row[1]).ljust(10)
              + str(row[2]).rjust(20)
              + str(row[3]).rjust(25)
              + str(row[4]).rjust(20)
              + str(row[5]).rjust(20)
        )


if __name__ == "__main__":
    stats()
