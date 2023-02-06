import os
import xml.etree.ElementTree as ET
import cv2
import numpy as np
import shutil
import pickle
from tqdm import tqdm

# The script should be importable but also executable from the terminal...
if __name__ == '__main__':
    import common
else:
    from . import common


detrac_classes_map = { # TODO
    "car":    common.classes.index("passenger_car"),
    "bus":    common.classes.index("bus"),
    "van":    common.classes.index("transporter"),
    "others": common.classes.index("truck")
}

def boxToBbox(box_elem):
    """Converts XML <box> element to [x1, y1, x2, y2] (in pixels)
    """
    x1 = float(box_elem.attrib["left"])
    x2 = x1 + float(box_elem.attrib["width"])
    y1 = float(box_elem.attrib["top"])
    y2 = y1 + float(box_elem.attrib["height"])
    return [round(p) for p in [x1, y1, x2, y2]]


def process_detrac():
    """Converts ground truth data of the DETRAC dataset from XML to
    mmdetection's middle format in a pickle file. Also combines all images to
    a separate folder while masking ignored areas

    This can take some time since the dataset is huge

    DETRAC format:
    <sequence name="MVI_20011">
        <sequence_attribute camera_state="unstable" sence_weather="sunny"/>
        <ignored_region>
            <box left="778.75" top="24.75" width="181.75" height="63.5"/>
            <box left="930.75" top="94.75" width="29.75" height="33.5"/>
        </ignored_region>
        <frame density="7" num="1"> # Density = amount of vehicles, num = image number
            <target_list>
                <target id="1">
                    <box left="592.75" top="378.8" width="160.05" height="162.2"/>
                    <attribute orientation="18.488" speed="6.859" trajectory_length="5" truncation_ratio="0.1" vehicle_type="car"/>
                </target>
                <target id="2">
                    <box left="557.65" top="120.98" width="47.2" height="43.06"/>
                    <attribute orientation="19.398" speed="1.5055" trajectory_length="72" truncation_ratio="0" vehicle_type="car"/>
                </target>
    ...

    mmdetection format:
    ```json
    [
        {
            'filename': 'a.jpg',
            'width': 1280,
            'height': 720,
            'ann': {
                'bboxes': <np.ndarray> (n, 4) in (x1, y1, x2, y2) order (values are in pixels),
                'labels': <np.ndarray> (n, ),
                'bboxes_ignore': <np.ndarray> (k, 4), (optional field)
                'labels_ignore': <np.ndarray> (k, 4) (optional field)
            }
        },
        ...
    ]
    ```
    """


    # Initialize paths
    # dataset_path = common.datasets["mio-tcd"]["path"]
    dataset_path = os.path.join(common.datasets_path, common.datasets["detrac"]["path"])
    dirnames = {
        "gt": {
            "train": os.path.join(dataset_path, "DETRAC-Train-Annotations-XML/"),
            "test": os.path.join(dataset_path, "DETRAC-Test-Annotations-XML/"),
        },
        "imgs":{
            "train": os.path.join(dataset_path, "Insight-MVT_Annotation_Train/"),
            "test": os.path.join(dataset_path, "Insight-MVT_Annotation_Test/"),
        },
    }

    gt_pickle_filepath = os.path.join(dataset_path, common.gt_pickle_filename)

    imgs_combined_dirname = "imgs_combined"
    imgs_combined_dirpath = os.path.join(dataset_path, imgs_combined_dirname)
    if os.path.exists(imgs_combined_dirpath):
        shutil.rmtree(imgs_combined_dirpath)
    os.mkdir(imgs_combined_dirpath)

    sequences = [] # Absolute paths to .xml (ground truth) files
    for key in list(dirnames["gt"].keys()):
        # Get filenames:
        sequences_tmp = os.listdir(dirnames["gt"][key])
        for seq in sequences_tmp: # For each filename
            # Save absolute path:
            sequences.append(os.path.join(dirnames["gt"][key], seq))

    # Remove ignored sequences
    for seq_filepath in sequences.copy():
        for seq_id in common.datasets["detrac"]["ignored_sequences"]:
            if seq_id in seq_filepath:
                sequences.remove(seq_filepath)

    print(f"Reading and processing DETRAC dataset: {len(sequences)} sequences")

    data_dict = {}
    for seq in tqdm(sequences):

        tree = ET.parse(seq)
        root = tree.getroot()

        subset = "train" if dirnames["gt"]["train"] in seq else "test"
        seq_name = root.attrib["name"]

        # Get ignored regions first
        ignored_regions = [] # [[x1, y1, x2, y2], ...]
        for child in root: # sequence_attribute, ignored_region, frame(s)
            if child.tag == "ignored_region":
                for box in child:
                    ignored_regions.append(boxToBbox(box))

        # Only then, read the annotations
        for child in tqdm(root, leave=False): # sequence_attribute, ignored_region, frame(s)

            # For each frame:
            if child.tag == "frame":
                frame_number = int(child.attrib["num"])
                img_filename = "img" + str(frame_number).zfill(5) + ".jpg"
                img_abs_filepath = os.path.join(dirnames["imgs"][subset], seq_name, img_filename) # Absolute

                # New filename will be {sequence_id}_{img_filename}
                new_img_filename = seq_name + "_" + img_filename
                new_img_rel_filepath = os.path.join(imgs_combined_dirname, new_img_filename)

                # Initialize the object in `data_dict` var if it doesn't exist yet
                if new_img_rel_filepath not in data_dict.keys():

                    # Get width and height of the image
                    height, width, _ = cv2.imread(img_abs_filepath).shape

                    data_dict[new_img_rel_filepath] = {
                        "filename": new_img_rel_filepath,
                        "width": int(width),
                        "height": int(height),
                        "ann": {
                            "bboxes": [],
                            "labels": []
                        }
                    }

                # Mask the ignored regions and save to imgs_combined folder
                # Read the image
                frame = cv2.imread(img_abs_filepath)
                # Draw a black rectangle for each region
                for region in ignored_regions:
                    start = (region[0], region[1])
                    end = (region[2], region[3])
                    frame = cv2.rectangle(frame, start, end, (0, 0, 0), -1)
                # Save the image to imgs_combined folder
                cv2.imwrite(os.path.join(dataset_path, new_img_rel_filepath), frame)

                # Get annotations and save them to data_dict
                for target_list in child:
                    for target in target_list:
                        for target_data in target:
                            if target_data.tag == "box":
                                bbox = boxToBbox(target_data)
                                data_dict[new_img_rel_filepath]["ann"]["bboxes"].append(bbox)
                            if target_data.tag == "attribute":
                                label = target_data.attrib["vehicle_type"]
                                label_mapped = detrac_classes_map[label]
                                data_dict[new_img_rel_filepath]["ann"]["labels"].append(label_mapped)

    # Convert data_dict to a list and all lists (bboxes and labels) to numpy arrays
    data_list = []
    print("Converting to numpy arrays")
    for key in list(data_dict.keys()):
        val = data_dict[key]

        # Convert lists of bboxes and labels to arrays
        # Should work if done the same way as labels, but to be sure..:
        val["ann"]["bboxes"] = np.array(
            [np.array(l) for l in val["ann"]["bboxes"]], 
            dtype=np.int16)
        val["ann"]["labels"] = np.array(val["ann"]["labels"], dtype=np.int16)

        data_list.append(val)

    print(f"Images: {len(data_list)}")
    annotations = sum([len(img["ann"]["labels"]) for img in data_list])
    print(f"Annotations: {annotations}")

    # Write the list to a file
    with open(gt_pickle_filepath, 'wb') as f:
        pickle.dump(data_list, f, protocol=common.pickle_file_protocol)

    print(f"Saved to {gt_pickle_filepath}")


if __name__ == "__main__":
    process_detrac()