"""
Converts ground truth data of the DETRAC dataset from XML to COCO format.

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
import os
import xml.etree.ElementTree as ET
from PIL import Image
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


detrac_classes_map = {
    "car":    common.classes_ids["car"],
    "bus":    common.classes_ids["bus"],
    "van":    common.classes_ids["transporter"],
    "others": common.classes_ids["truck"]
}

# Initialize paths
rel_dirpaths = {
    "gt": {
        "train": "DETRAC-Train-Annotations-XML/",
        "test": "DETRAC-Test-Annotations-XML/",
    },
    "imgs":{
        "train": "Insight-MVT_Annotation_Train/",
        "test": "Insight-MVT_Annotation_Test/",
    },
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

    abs_dirpaths = {}
    for key in rel_dirpaths:
        abs_dirpaths[key] = {}
        for subset in rel_dirpaths[key]:
            abs_dirpaths[key][subset] = os.path.join(
                common.paths.datasets_dirpath, 
                common.datasets["detrac"]["path"],
                rel_dirpaths[key][subset])


    print("Getting a list of sequences")
    sequences = [] # Absolute paths to .xml (ground truth) files
    for key in abs_dirpaths["gt"]:
        # Get filenames:
        sequences_tmp = os.listdir(abs_dirpaths["gt"][key])
        for seq in sequences_tmp: # For each filename
            # Save absolute path:
            sequences.append(os.path.join(abs_dirpaths["gt"][key], seq))

    # print("Removing ignored sequences") # No need after reannotation in labelbox
    # for seq_filepath in sequences.copy():
    #     for seq_id in common.datasets["detrac"]["ignored_sequences"]:
    #         if seq_id in seq_filepath:
    #             sequences.remove(seq_filepath)

    data = {
        "images": [],
        "annotations": [],
    }

    anno_id_counter = 0
    img_id_counter = 0
    mask_obj_id_counter = 990000

    print(f"Reading and processing sequences of the DETRAC dataset")
    for seq in tqdm(sequences):

        tree = ET.parse(seq)
        root = tree.getroot()

        subset = "train" if rel_dirpaths["gt"]["train"] in seq else "test"
        seq_name = root.attrib["name"]

        # Get ignored regions first
        ignored_regions = [] # [[x, y, w, h], ...]
        for child in root: # sequence_attribute, ignored_region, frame(s)
            if child.tag == "ignored_region":
                for box in child:
                    ignored_regions.append(boxToBbox(box))

        # Only then, read the annotations
        for child in root: # sequence_attribute, ignored_region, frame(s)

            # For each frame:
            if child.tag == "frame":

                frame_nr = int(child.attrib["num"])
                img_id = img_id_counter

                img_filename = "img" + str(frame_nr).zfill(5) + ".jpg"
                img_rel_filepath = os.path.join(rel_dirpaths["imgs"][subset],
                                                seq_name,
                                                img_filename)
                img_abs_filepath = os.path.join(abs_dirpaths["imgs"][subset], 
                                                seq_name,
                                                img_filename)

                # Get width and height of the image
                width, height = Image.open(img_abs_filepath).size

                # Append to data["images"]
                data["images"].append({
                    "id": img_id,
                    "file_name": img_rel_filepath,
                    "height": height,
                    "width": width,
                    "frame": frame_nr
                })

                # Append masks to data["annotations"]
                for i in range(len(ignored_regions)):
                    data["annotations"].append({
                        "id": anno_id_counter,
                        "image_id": img_id,
                        "category_id": common.classes_ids["mask"],
                        "bbox": ignored_regions[i],
                        "object_id": mask_obj_id_counter + i
                    })

                    anno_id_counter += 1

                # Get annotations and save them
                for target_list in child:
                    for target in target_list:

                        id = target.attrib["id"]
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
                            "bbox": bbox,
                            "object_id": id
                        })

                        anno_id_counter += 1

                img_id_counter += 1

        mask_obj_id_counter += len(ignored_regions)

    common.save_processed("detrac", data)


if __name__ == "__main__":
    # import cProfile
    # import pstats
    # pr = cProfile.Profile()
    # pr.enable()
    # process_detrac()
    # pr.disable()
    # ps = pstats.Stats(pr)
    # ps.sort_stats('cumtime')
    # ps.print_stats(10)

    process_detrac()
