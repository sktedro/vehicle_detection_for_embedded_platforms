import os
import pickle

# The script should be importable but also executable from the terminal...
if __name__ == '__main__':
    import common
else:
    from . import common


def printClassCounts():
    classes_counts = [0] * len(common.classes)
    images_count = 0
    with open(common.dataset_pickle_filepath, "rb") as f:
        data = pickle.load(f)
        for key in ["train", "val", "test"]:
            for img in data[key]:
                images_count += 1
                for label in img["ann"]["labels"]:
                    classes_counts[label] += 1

    print("Class instances in all images:")

    print("==================================================")

    for i in range(len(classes_counts)):
        print(f"{common.classes_dict[i]} ({i}):".ljust(40) + str(classes_counts[i]).rjust(10))

    print("==================================================")

    print("Number of images:".ljust(40) + str(images_count).rjust(10))
    print("Number of instances:".ljust(40) + str(sum(classes_counts)).rjust(10))


if __name__ == "__main__":
    printClassCounts()