"""
Takes the entire dataset and copies all necessary data for a certain subset
to a new directory - leaves out data that are not part of the subset.
This might be useful when you only need testing data on an embedded device.
Subset names (train, val, test) are required as arguments
"""

import json
import sys
import os
import shutil
from tqdm import tqdm

# The script should be importable but also executable from the terminal...
if __name__ == '__main__':
    import common
else:
    try:
        from . import common
    except:
        import sys
        sys.path.append(os.path.dirname(__file__))
        import common


def copy_dataset_subsets(subset_names):
    for subset_name in subset_names:
        assert subset_name in ["train", "val", "test"], "Subset names can be: train, val, test"

    # Create the output directory (remove existing if exists)
    out_dirname = "dataset_" + "_".join(subset_names)
    out_dirpath = os.path.join(common.paths.datasets_dirpath, out_dirname)
    if os.path.exists(out_dirpath):
        shutil.rmtree(out_dirpath)
    os.mkdir(out_dirpath)

    for subset_name in subset_names:
        print(f"Copying {subset_name} subset")

        # Copy the gt file
        gt_filepath = os.path.join(
            common.paths.datasets_dirpath,
            common.gt_combined_filenames[subset_name])
        new_gt_filepath = os.path.join(
            out_dirpath,
            common.gt_combined_filenames[subset_name])
        shutil.copyfile(gt_filepath, new_gt_filepath)

        # Load dataset subset
        with open(gt_filepath) as f:
            data = json.load(f)

        # Copy each image of the subset to the out_dirpath
        for img in tqdm(data["images"]):
            img_rel_filepath = img["file_name"]
            img_abs_filepath = os.path.join(common.paths.datasets_dirpath, img_rel_filepath)
            new_img_abs_filepath = os.path.join(out_dirpath, img_rel_filepath)

            # Create the path to the image if it doesn't exist
            new_img_abs_dirpath = os.path.dirname(new_img_abs_filepath)
            if not os.path.exists(new_img_abs_dirpath):
                os.makedirs(new_img_abs_dirpath, exist_ok=True)

            # Copy the image
            shutil.copyfile(img_abs_filepath, new_img_abs_filepath)

    print("All done")


if __name__ == "__main__":
    copy_dataset_subsets(sys.argv[1:])
