import json

# The script should be importable but also executable from the terminal...
if __name__ == '__main__':
    import common
else:
    from . import common

def printClassCounts():
    classes_counts = [0] * len(common.classes_ids)
    with open(common.dataset_filepath) as f:
        data = json.loads(f.read())
        images_count = len(data["images"])
        for anno in data["annotations"]:
            classes_counts[anno["category_id"] - 1] += 1

    print("Class instances in all images:")

    print("==================================================")

    for i in range(len(classes_counts)): # Class id starts at 1
        print(f"{common.classes_names[i + 1]} ({i + 1}):".ljust(40) + str(classes_counts[i]).rjust(10))

    print("==================================================")

    print("Number of images:".ljust(40) + str(images_count).rjust(10))
    print("Number of instances:".ljust(40) + str(sum(classes_counts)).rjust(10))


if __name__ == "__main__":
    printClassCounts()
