"""
Uploads the DETRAC dataset (processed to COCO) to labelbox (API key shall be
provided as an argument). Videos (name example MVI_40212) should be uploaded (to
dataset "detrac") before running this script and ontology should be created
beforehand, too (ID should be provided in `ontology_id`).

Script creates a project if it doesn't yet exist, uses an existing "detrac"
dataset where for each data row, it uploads annotations as Model Assisted
Labeling Predictions.
"""
import os
import sys
import json
import uuid
import labelbox as lb
from tqdm import tqdm

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import common


ontology_id = "cleldglxk6f3u07za15bbfide"

def detrac_to_labelbox(api_key):

    # Initialize paths
    dataset_abs_dirpath = os.path.join(common.paths.datasets_dirpath, common.datasets["detrac"]["path"])

    with open(os.path.join(dataset_abs_dirpath, common.gt_unmasked_filenames["combined"])) as f:
        gt = json.load(f)

    client = lb.Client(api_key=api_key)

    # Get the project
    projects = client.get_projects()
    for project in projects:
        if project.name == "detrac":
            break
    else: # If project was not found
        project = client.create_project(
            name="detrac",
            media_type=lb.MediaType.Video)
        ontology = client.get_ontology(ontology_id)
        project.setup_editor(ontology)
    # print(ontology)


    datasets = client.get_datasets()
    for dataset in datasets:
        if dataset.name == "detrac":
            break
    if dataset == None or dataset.name != "detrac":
        raise Exception("Dataset not found")
        
    # Process annotations efficiently
    annotations_in_img = {} # Key is IMG ID, val is an annotation
    for anno in gt["annotations"]:
        img_id = anno["image_id"]
        if img_id not in annotations_in_img:
            annotations_in_img[img_id] = []
        annotations_in_img[img_id].append(anno)

    for data_row in tqdm(dataset.data_rows()):
        video_filename = data_row.external_id
        video_name = video_filename.split(".")[0]
        # print(data_row)

        # Check if the batch doesn't already exist
        batch_name = "batch_" + video_name
        found = False
        for batch in project.batches():
            if batch.name == batch_name:
                found = True
        if found:
            print(f"Batch {batch_name} found. Skipping")
            continue

        # Create a new batch
        batch = project.create_batch(
            name=batch_name,
            data_rows=[data_row],
            priority=1
        )

        # Get all images for this sequence
        imgs = {} # Key = img_id, val = img (as in COCO)
        for img in gt["images"].copy():
            if video_name in img["file_name"]:
                imgs[img["id"]] = img
                gt["images"].remove(img) # To optimize

        # Get all annotations for this sequence
        annos = []
        for img_id in imgs:
            annos += annotations_in_img[img_id]

        # Aggregate: per object id, then by frame number
        aggr = {}
        for anno in annos:
            obj_id = anno["object_id"]
            frame_nr = imgs[anno["image_id"]]["frame"]
            if obj_id not in aggr:
                aggr[obj_id] = {}
                aggr[obj_id]["category_id"] = anno["category_id"]
            if frame_nr not in aggr[obj_id]:
                aggr[obj_id][frame_nr] = anno

        labels = []
        for obj_id in aggr:
            cls = common.classes_names[aggr[obj_id]["category_id"]]
            obj_annos = []
            for frame_nr in [key for key in aggr[obj_id] if key != "category_id"]:
                x1, y1, w, h = aggr[obj_id][frame_nr]["bbox"]
                obj_annos.append({
                    "frame": frame_nr,
                    "bbox": {
                        "left": x1,
                        "top": y1,
                        "width": w,
                        "height": h,
                    }
                })
            labels.append({
                "name": cls,
                "segments": [
                    {
                        "keyframes": obj_annos
                    }
                ],
                "uuid": str(uuid.uuid4()), # Cannot be obj_id
                "dataRow": {
                    "id": data_row.uid
                }
            })

        # labels = [
        #     {
        #         'name': 'car',
        #         'dataRow': {
        #             # 'id': 'clemt0t0j2whx070t0xel108n'
        #             'id': data_row.uid
        #             },
        #         'segments': [
        #             {
        #                 'keyframes': [
        #                     {
        #                         'bbox': {
        #                             'height': 330,
        #                             'left': 100,
        #                             'top': 100,
        #                             'width': 225
        #                             },
        #                         'frame': 10
        #                         }
        #                     ]
        #                 }
        #             ],
        #         'uuid': '97293ce1-faa5-4f9d-ba1c-ace92be8c11a'
        #         }
        #     ]

        # for label in labels:
        #     assert list(label.keys()) == ["name", "segments", "uuid", "dataRow"]

        #     assert isinstance(label["name"], str)
        #     assert label["name"] in list(common.classes_ids.keys())

        #     assert isinstance(label["segments"], list)
        #     for segment in label["segments"]:

        #         assert isinstance(segment, dict)
        #         assert list(segment.keys()) == ["keyframes"]
        #         for keyframe in segment["keyframes"]:

        #             assert isinstance(keyframe, dict)
        #             assert list(keyframe.keys()) == ["frame", "bbox"]

        #             assert isinstance(keyframe["frame"], int)
        #             assert isinstance(keyframe["bbox"], dict)
        #             for dim in keyframe["bbox"]:
        #                 assert isinstance(keyframe["bbox"][dim], int)

        #     assert isinstance(label["uuid"], str)

        #     assert isinstance(label["dataRow"], dict)
        #     assert list(label["dataRow"].keys()) == ["id"]
        #     assert label["dataRow"]["id"] == data_row.uid

        # Upload MAL label for this data row in project
        upload_job_mal = lb.MALPredictionImport.create_from_objects(
            client = client,
            project_id = project.uid,
            name="mal_import_" + str(uuid.uuid4()), 
            predictions=labels)

        upload_job_mal.wait_until_done()
        if upload_job_mal.errors != []:
            raise Exception("Errors:", upload_job_mal.errors)


if __name__ == "__main__":
    detrac_to_labelbox(sys.argv[1])
