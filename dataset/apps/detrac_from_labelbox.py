"""
Downloads annotations from labelbox, converts them to COCO format and saves
them. API key should be provided as an argument and project ID in `proj_id`
variable.
"""
import os
import labelbox
import ndjson
import requests
import sys
from tqdm import tqdm

try:
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
except:
    sys.path.append(os.path.join(os.path.dirname("__file__"), '..'))
import common
import detrac

proj_id = 'clephuroo2vg407y0fbascqa1'


def detrac_from_labelbox(api_key):

    lb = labelbox.Client(api_key=api_key)
    project = lb.get_project(proj_id)


    """
    <DataRow {
        "created_at": "2023-02-28 10:03:57+00:00",
        "external_id": "MVI_29134.mp4",
        "global_key": null,
        "media_attributes": {
            "codec": "avc1",
            "width": 960,
            "height": 540,
            "codecTag": "0x31637661",
            "duration": 32,
            "mimeType": "video/mp4",
            "codecName": "h264",
            "frameRate": 25,
            "frameCount": 800,
            "contentLength": 6232640,
            "sampleAspectRatio": "1:1"
        },
        "metadata": [],
        "metadata_fields": [],
        "row_data": "https://storage.labelbox.com/clb5hwp0c0wha07yk99yiej51%2Fc2992233-f6a9-ec5d-c2f3-6bfb705adac9-MVI_29134.mp4?Expires=1677753948492&KeyName=labelbox-assets-key-3&Signature=-ZmUHe8IJDpxfMcOIDvyxpnB650",
        "uid": "cleo308zqobrb074b3kdh6dwc",
        "updated_at": "2023-02-28 10:03:57+00:00"
    }>
    """

    # Create a dictionary where {key=data_row_id : value=data_row_info}
    # Note: this fetches all sequences, not just labeled (done) ones
    print("Fetching data rows")
    sequences = {}
    subsets = list(project.batches()) if len(list(project.batches())) > 0 else list(project.datasets())
    for subset in tqdm(subsets):
        for data_row in subset.export_data_rows():
            sequences.update({data_row.external_id: {
                "width": data_row.media_attributes["width"],
                "height": data_row.media_attributes["height"],
                "frames": data_row.media_attributes["frameCount"],
                "sequence_name": data_row.external_id.split(".")[0]
            }})


    """
    Export = (only showing interesting keys)
    {
    'DataRow ID': 'cleo308zvobxf074b335pcyt9',
    'Dataset Name': 'detrac',
    'External ID': 'MVI_63563.mp4',
    'ID': 'clephvf0g0ga6070c3dt63b41',
    'Label': {'classifications': [],
            'frames': 'https://api.labelbox.com/v1/frames/clephvf0g0ga604470c3dt63b41sd3214',
            'objects': [],
            'relationships': []},
    'Labeled Data': 'https://storage.labelbox.com/clb5hwp0c0wha07yk99yiej51%2F1fe7c2140-3715-a3c8-1a5a-f2c8767ac82e8-MVI_63563.mp4',
    'Project Name': 'detrac',
    }
    """

    """
    Annotations =
    [{
    'frameNumber': 1,
    'objects': [{'bbox': {'height': 142, 'left': 614, 'top': 338, 'width': 225},
                'featureId': 'clephvgby2rip07z2dxvk50xo',
                'keyframe': True,
                'value': 'car'},
                ...]
    }]

    """

    #%%
    print("Fetching labels")

    data = {"images": [],
            "annotations": []}
    exports = project.export_labels(download=True)
    img_id_counter = 0
    anno_id_counter = 0
    for export in tqdm(exports):
        file_name = export["External ID"]
        sequence = sequences[file_name]
        seq_name = sequence["sequence_name"]

        annotations_url = export["Label"]["frames"]
        headers = {"Authorization": f"Bearer {api_key}"}
        annotations = ndjson.loads(requests.get(annotations_url, headers=headers).text)

        for img_annotations in annotations:
            frame_nr = img_annotations["frameNumber"]

            img_filename = "img" + str(frame_nr).zfill(5) + ".jpg"
            img_dirname = seq_name

            # Get the image filepath by searching for it (it can be in the train
            # subset's directory or in test subset's dir)
            if os.path.exists(os.path.join(common.paths.datasets_dirpath,
                                        common.datasets["detrac"]["path"],
                                        detrac.rel_dirpaths["imgs"]["train"],
                                        img_dirname,
                                        img_filename)):
                img_subset_dirname = detrac.rel_dirpaths["imgs"]["train"]
            elif os.path.exists(os.path.join(common.paths.datasets_dirpath,
                                            common.datasets["detrac"]["path"],
                                            detrac.rel_dirpaths["imgs"]["test"],
                                            img_dirname,
                                            img_filename)):
                img_subset_dirname = detrac.rel_dirpaths["imgs"]["test"]
            else:
                raise Exception(f"File {img_dirname}/{img_filename} not found")

            img_rel_filepath = os.path.join(img_subset_dirname,
                                            img_dirname,
                                            img_filename)

            data["images"].append({
                "id": img_id_counter,
                "frame": frame_nr,
                "width": sequence["width"],
                "height": sequence["height"],
                "file_name": img_rel_filepath
            })

            for anno in img_annotations["objects"]:
                bbox = anno["bbox"]
                data["annotations"].append({
                    "id": anno_id_counter,
                    "image_id": img_id_counter,
                    "category_id": common.classes_ids[anno["value"]],
                    "bbox": [bbox["left"], bbox["top"], bbox["width"], bbox["height"]],
                })
                anno_id_counter += 1

            img_id_counter += 1

    print("Saving data")
    common.save_processed("detrac", data)


if __name__ == "__main__":
    detrac_from_labelbox(sys.argv[1])
