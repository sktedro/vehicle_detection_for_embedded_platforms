import os
import xml.etree.ElementTree as ET
import cv2
import shutil
from tqdm import tqdm

# The script should be importable but also executable from the terminal...
if __name__ == '__main__':
    import common
else:
    from . import common


detrac_classes_map = {
    "car":    common.classes_ids["car"],
    "bus":    common.classes_ids["bus"],
    "van":    common.classes_ids["transporter"],
    "others": common.classes_ids["truck"]
}

def boxToBbox(box_elem):
    """Converts XML <box> element to [x1, y1, w, h] (in pixels)
    """
    x1 = float(box_elem.attrib["left"])
    y1 = float(box_elem.attrib["top"])
    w = float(box_elem.attrib["width"])
    h = float(box_elem.attrib["height"])
    return [round(p) for p in [x1, y1, w, h]]


def process_detrac():
    """Converts ground truth data of the DETRAC dataset from XML to COCO format.
    Also combines all images to a separate folder while masking ignored areas

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
    """

    # Initialize paths
    dataset_abs_dirpath = os.path.join(common.datasets_dirpath, common.datasets["detrac"]["path"])
    abs_dirpaths = {
        "gt": {
            "train": os.path.join(dataset_abs_dirpath, "DETRAC-Train-Annotations-XML/"),
            "test": os.path.join(dataset_abs_dirpath, "DETRAC-Test-Annotations-XML/"),
        },
        "imgs":{
            "train": os.path.join(dataset_abs_dirpath, "Insight-MVT_Annotation_Train/"),
            "test": os.path.join(dataset_abs_dirpath, "Insight-MVT_Annotation_Test/"),
        },
    }

    # Create a directory for combined images (delete it first if exists)
    imgs_combined_dirname = "imgs_combined"
    imgs_combined_abs_dirpath = os.path.join(dataset_abs_dirpath, imgs_combined_dirname)
    if os.path.exists(imgs_combined_abs_dirpath):
        shutil.rmtree(imgs_combined_abs_dirpath)
    os.mkdir(imgs_combined_abs_dirpath)

    print("Getting a list of sequences")
    sequences = [] # Absolute paths to .xml (ground truth) files
    for key in tqdm(list(abs_dirpaths["gt"].keys())):
        # Get filenames:
        sequences_tmp = os.listdir(abs_dirpaths["gt"][key])
        for seq in sequences_tmp: # For each filename
            # Save absolute path:
            sequences.append(os.path.join(abs_dirpaths["gt"][key], seq))

    print("Removing ignored sequences")
    for seq_filepath in tqdm(sequences.copy()):
        for seq_id in common.datasets["detrac"]["ignored_sequences"]:
            if seq_id in seq_filepath:
                sequences.remove(seq_filepath)

    data = {
        "images": [],
        "annotations": []
    }

    anno_id_counter = 0
    img_id_counter = 0

    print(f"Reading and processing sequences of the DETRAC dataset")
    for seq in tqdm(sequences):

        tree = ET.parse(seq)
        root = tree.getroot()

        subset = "train" if abs_dirpaths["gt"]["train"] in seq else "test"
        seq_name = root.attrib["name"]

        # Get ignored regions first
        ignored_regions = [] # [[x1, y1, x2, y2], ...]
        for child in root: # sequence_attribute, ignored_region, frame(s)
            if child.tag == "ignored_region":
                for box in child:
                    # Get ignored region as a list and convert x,y,w,h to x1,y2,x2,y2
                    bbox = boxToBbox(box)
                    bbox[2] += bbox[0]
                    bbox[3] += bbox[1]
                    ignored_regions.append(bbox)

        # Only then, read the annotations
        for child in tqdm(root, leave=False): # sequence_attribute, ignored_region, frame(s)

            # For each frame:
            if child.tag == "frame":

                img_number = int(child.attrib["num"])
                img_id = img_id_counter

                old_img_filename = "img" + str(img_number).zfill(5) + ".jpg"
                old_img_abs_filepath = os.path.join(abs_dirpaths["imgs"][subset], seq_name, old_img_filename)

                new_img_filename = seq_name + "_" + str(img_id_counter).zfill(6) + ".jpg"
                new_img_rel_filepath = os.path.join(common.datasets["detrac"]["path"], imgs_combined_dirname, new_img_filename)

                # Mask the ignored regions and save to imgs_combined folder
                frame = cv2.imread(old_img_abs_filepath)
                # Draw a black rectangle for each region
                for region in ignored_regions:
                    start = (region[0], region[1])
                    end = (region[2], region[3])
                    frame = cv2.rectangle(frame, start, end, (0, 0, 0), -1)
                # Save the image to imgs_combined folder
                cv2.imwrite(os.path.join(common.datasets_dirpath, new_img_rel_filepath), frame)

                # Append to data["images"]
                if new_img_rel_filepath not in data.keys():

                    # Get width and height of the image
                    height, width, _ = cv2.imread(old_img_abs_filepath).shape

                    data["images"].append({
                        "id": img_id,
                        "file_name": new_img_rel_filepath,
                        "height": height,
                        "width": width
                    })

                # Get annotations and save them
                for target_list in child:
                    for target in target_list:

                        bbox = None
                        label = None
                        for target_data in target:
                            if target_data.tag == "box":
                                bbox = boxToBbox(target_data)
                            if target_data.tag == "attribute":
                                label = target_data.attrib["vehicle_type"]

                        data["annotations"].append({
                            "id": anno_id_counter,
                            "image_id": img_id,
                            "category_id": detrac_classes_map[label],
                            "bbox": bbox
                        })

                        anno_id_counter += 1
                
                img_id_counter += 1
    
    common.save_processed("detrac", data)


if __name__ == "__main__":
    process_detrac()