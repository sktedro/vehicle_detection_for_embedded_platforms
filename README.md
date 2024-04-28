# Structure of This README

- Warning
- Brief overview
- Not so brief overview
- Results
- Project structure (folders and files with brief explaination)
- Datasets (detailed)
- Models FLOPS and params table
- Testing (how to, and the format of the output)
- Requirements
- FAQ and Frequent Errors
- Installation on NVIDIA Jetson


# Warning

This code is a bit of a mess. No time was left for refactoring, so some
solutions are ugly, some scripts might only work if executed from their
directory or from the main directory. Also, `update_paths.sh` needs to be used
because some files require some paths in them (like, `paths.py` needs the path
of datasets). Or the user could do it manually if he doesn't trust the script
(it asks before doing any changes).

Most python scripts use argparse, so `--help` displays options. However, older
scripts (like ones in `dataset/`) usually don't!


# Brief

This project focuses on benchmarking YOLOv8 object detectors on traffic images
of a surveillance type. Models of different sizes (including non-standard
YOLOv8-pico and YOLOv8-femto) with different input resolutions can be tested
with different backends (PyTorch, ONNX Runtime, TensorRT) on different devices.
When deploying to TensorRT, quantization to FP16 and INT8 can be done.


# Not so Brief

This project heavily utilized libraries of the OpenMMLab project, including
MMYOLO, MMDetection and MMDeploy. For specific versions, see `requirements.txt`.
Newer versions of MMYOLO should work.

Thirteen different models were trained:
- YOLOv8-medium 640x384
- YOLOv8-small 640x384
- YOLOv8-large with MobileNetV2 backbone 512x288 (indices used: [2, 4, 6])
- YOLOv8-nano 640x384
- YOLOv8-nano 512x288
- YOLOv8-nano 448x256
- YOLOv8-pico 512x288
- YOLOv8-pico 448x256
- YOLOv8-pico 384x224
- YOLOv8-femto 512x288
- YOLOv8-femto 448x256
- YOLOv8-femto 384x224
- YOLOv8-femto 352x192

Six datasets are used: UA-DETRAC (reannotated), MIO-TCD (test subset), AAU
RainSnow (RGB camera only), MTID (beware of unannotated frames), NDIS Park
(transporters and trailers were reannotated) and VisDrone (DET subset). Details
will be explained later. Scripts for processing the datasets (and combining
them) are provided.

When training standard YOLOv8 models (medium, small, nano) and using the
MobileNetV2 backbone, pre-trained models were used (found in MMYOLO and
MMPretrain (previously known as MMClassification) repositories).

All models can be deployed to both ONNX and TensorRT, as static or dynamic in
the batch size dimension (dynamic batch, static input resolution).  When
deploying to TensorRT, FP16 quantization and INT8 quantization is possible (INT8
requires weight calibration using the validation dataset).

Inference can be done with PyTorch, ONNX Runtime and TensorRT.

Six devices were benchmarked:
- NVIDIA Jetson AGX Xavier (TensorRT and ONNX Runtime; fp32, fp16, int8)
- NVIDIA Jetson Xavier NX (TensorRT and ONNX Runtime; fp32, fp16, int8)
- NVIDIA Jetson Nano (TensorRT and ONNX Runtime; fp32, fp16)
- NVIDIA MX150 (ONNX Runtime GPU)
- Intel Core i7-9850H (ONNX Runtime) - tests were run with a single CPU thread
- Raspberry Pi 4B (ONNX Runtime) - tests were run with a single CPU thread

Batch sizes used when benchmarking: 1, 2, 4, 8, 16, 32

Benchmark results contain tests of all models (all input resolutions) on all
devices with all batch sizes (exception is that some models did not fit into
memory with large batch sizes). See `Testing` chapter for details

We recommend reading the project structure before using this repository, because
the project is a bit of a mess. Or just don't use this repository.


# Results

How the models performed on the test set: https://youtube.com/playlist?list=PLt6xFMYrdS_r923yO-dadzoD1YIBi0zNL

Paper can be found in root as `paper.pdf` or at the publisher's website: https://www.vut.cz/www_base/zav_prace_soubor_verejne.php?file_id=252433.

Training checkpoints - .pth files (some of them) can be found here: https://drive.google.com/drive/folders/12dLInS5LaBuHbThuOophZli-DyCh7t9h?usp=sharing


# Project Structure

Not sorted by name!

```py
. # Repository folder
├── checkpoints/ # Where to put pre-trained models
│   ├── print_model_structure.py # Prints info about layers in a PyTorch model
│   └── example_pre-trained_model.pth
├── configs/ # MMYOLO (MMDetection) model configurations. The base is `YOLOv8_m.py`. Edit `settings.py` before training a specific model!
│   ├── settings.py # Contains additional config, such as `num_gpus`, `img_scale`, `max_epochs`, `base_lr`
│   ├── yolov8_f_352x192.py
│   ├── yolov8_f_384x224.py
│   ├── yolov8_f_448x256.py
│   ├── yolov8_f_512x288.py
│   ├── yolov8_l_mobilenet_v2_512x288_indices_246.py
│   ├── yolov8_m.py # Base
│   ├── yolov8_n_448x256.py
│   ├── yolov8_n_512x288.py
│   ├── yolov8_n_640x384.py
│   ├── yolov8_p_384x224.py
│   ├── yolov8_p_448x256.py
│   ├── yolov8_p_512x288.py
│   └── yolov8_s.py
├── dataset/ # Dataset processing scripts and such
│   ├── aau.py # Process AAU from their original format
│   ├── apply_masks.py # Apply masks to a dataset (or datasets)
│   ├── apps/ # Utilize applications - LabelBox and FiftyOne
│   │   ├── detrac_from_labelbox.py # Upload DETRAC dataset to LabelBox
│   │   ├── detrac_to_labelbox.py # Download DETRAC dataset from LabelBox
│   │   ├── fiftyone_create.py # Create a FiftyOne dataset
│   │   └── fiftyone_run.py # Run a FiftyOne server
│   ├── combine_datasets.py # Combine all datasets into one (json file)
│   ├── common.py # Defines classes, paths, shared functions and so on
│   └── copy_dataset_subsets.py # Copies only the necessary data (used images and files) to create a folder with a dataset subset (train, val, test; can be a combination of them) without unnecessary data, like unmasked images and such
│   ├── detrac.py # Process DETRAC from original format (useless after reannotation in LabelBox)
│   ├── __init__.py
│   ├── mio-tcd.py # Process MIO-TCD
│   ├── mtid.py # Process MTID
│   ├── ndis.py # Process NDIS Park
│   ├── split_dataset_into_subsets.py # Split the combined dataset into train, val, test
│   ├── split_test_by_dataset.py # Split the test subset by origin dataset (in our case, to DETRAC and MIO-TCD parts)
│   ├── stats.py # Counts class instances and number of images in the dataset files (combined, train, val, test)
│   ├── visdrone_det.py # Process VisDrone DET
│   ├── visualize_dataset.py # Visualize a processed dataset by drawing bboxes and class names on images
│   └── visualized_to_video.py # Convert visualized dataset (images with bboxes and classes) to videos (one folder -> one video)
├── deploy/ # Scripts and configs for model deployment
│   ├── common.py # To replace PLACEHOLDERS in deploy configs by specific values for a specific model. Needs to be used to build a deploy config!
│   ├── config_onnxruntime_dynamic_batch.py # ONNX Runtime deploy config - dynamic batch size
│   ├── config_onnxruntime_static.py # ONNX Runtime deploy config - static shape
│   ├── config_tensorrt_dynamic_batch_fp16.py # ...
│   ├── config_tensorrt_dynamic_batch_int8.py
│   ├── config_tensorrt_dynamic_batch.py
│   ├── config_tensorrt_static.py
│   ├── demo.jpg # Demo image used when deploying
│   ├── deploy_all.py # Deploy a model for all work dirs in the project directory with all deploy configs in this directory
│   ├── deploy_model.py # Deploy a single model
│   └── README.md
├── doc/ # The paper and the source code for the paper
│   └── doc.pdf # The paper...
├── experiments/ # Files generating the plots and tables used in the paper. They are ugly! Don't look at them!!!
│   └── # DO NOT OPEN THIS FOLDER! (because every single line is ugly)
├── save_stream/ # Automatically record RTSP or HLS streams. Not well documented because in the end, it was not used.
│   ├── hls/ # Record HLS streams (m3u8)
│   │   ├── config_example # Example of a config file. Can contain more entries
│   │   ├── process_files.sh # I guess it compresses and maybe uploads the files to cloud?
│   │   ├── README.md
│   │   └── save_stream.py # Save stream specified in the config file
│   ├── mount_and_compress.py # Mounts a cloud folder and compresses all uncompressed files in the folder?
│   └── rtsp/ # Record RTSP streams
│       ├── auto.sh # Do everything automatically
│       ├── process_files.sh
│       ├── README.md
│       └── save_stream.py
├── tests/ # Results of tests, their and deploy logs and so on..
│   ├── device_name/ # Not provided in repository!
│   │   ├── work_dir_name/
│   │   │   └── test_tensorrt_dynamic_batch_engine_batch16.log # example
│   │   ├── test_script_logs...
│   │   ├── deploy_script_logs...
│   │   └── test_all_results.json # Containing the complete test results from the device
│   ├── mAP/ # Tests which were run just for mAP values. For whole test dataset, for DETRAC only and for MIO-TCD only. With: i7, onnxruntime, dynamic, batch 8, "none" quantization
│   │   ├── test_all_mAP_results.json
│   │   ├── test_detrac_mAP_results.json
│   │   └── test_mio-tcd_mAP_results.json
│   ├── test_results.json # Merged test results from all devices
│   ├── models_analysis_log.txt # Containing the log of getting FLOPS and number of params of all models
│   └── tensorflow_dump.png # Validation mAP metrics when training all models. It's a mess!
├── vid/ # Sample videos used, for example, for inference
│   └── MVI_40701.mp4 # Example video we used to visually assess model performance (from DETRAC dataset)
├── working_dir_yolov8_f_512x288/ # Example working dir name
│   ├── best_coco/ # Contains a .pth file with the best epoch based on validation mAP (generated when training)
│   ├── coco_eval.json # File generated by extracting precision and recall from the test script
│   ├── epoch_10.pth # Example checkpoint file name
│   ├── last_checkpoint # Simple text file with absolute path to the last checkpoint, useful when continuing in a training session
│   └── yolov8_f_512x288.py # Model config exactly as was used when training. Contains absolute paths (probably)!

├── custom_modules.py # Contains the custom cut-out augmentation (it is registered by MMYOLO before training)

├── paths.py # Paths to datasets, OpenMMLab libs, checkpoints, work dirs... So very important!
├── update_paths.sh # A simple shell script to update absolute paths in relevant files using `sed`. It lists the files that might be affected and asks the user for confirmation. For example: `sh update_paths.sh "/home/tedro/Downloads/datasets" "/home/user/datasets" && sh update_paths.sh "/home/tedro/Desktop/d_projekty/bp/proj" "/home/user/repo"`
├── visualize_pipeline.py # Visualize a training/validation/test pipeline and for example, see augmented images

├── train.py # Training script. Before training, be sure to at least customize `paths.py`, `configs/yolov8_m.py` and `configs/settings.py`
├── dist_train.py # Training script for distributed training (N GPUs). Number of GPUs specified in `configs/settings.py`

├── inference_ort.py # Run inference using ONNX Runtime. Needs work but works
├── inference_pytorch.py # Run inference using PyTorch (using MMDet)
├── inference_trt.py # Run inference using TensorRT (using MMDet)

├── test_deployed.py # Run a test on a deployed model (using MMDeploy and the test dataset)
├── test_all.py # Run `test_all.py` for all available deployed models
├── collect_test_results.py # Iterates through all working dirs and reads their test logs to save them into `test_all_results.json`
├── merge_test_results.py # When test results are collected from different devices, each having a `test_all_results.json`, this script merges the files into `test_results.json` with all results in a single JSON

├── print_model_structure.py # Reads a config file to print a structure of a model
└── requirements.txt # Pip packages
```


# Datasets

- When training, they should be in COCO format (x,y,w,h format)
- UA-DETRAC was reannotated! Read its section for more info


### Output of `stats.py` for our specific datasets:
```
Number of images:                                                  
================================================================================
images_total                images_train          images_val         images_test
218821                            203061                7770                7990

Class instances in all images:
===================================================================================================================
class_name          class_id       instances_total          instances_train       instances_val      instances_test
bicycle             1                        14036                    13777                 115                 144
motorcycle          2                        17187                    16956                  97                 134
car                 3                       916317                   846331               29803               40183
transporter         4                       123585                   112614                4176                6795
bus                 5                        64529                    62890                 702                 937
truck               6                        38069                    36554                 712                 803
trailer             7                         2360                     2109                 106                 145
unknown             8                        28712                    26238                1258                1216
mask                9                            0                        0                   0                   0
sum                                        1204795                  1117469               36969               50357
```


### DETRAC

- https://detrac-db.rit.albany.edu/download
- Night, sunny, cloudy, rainy
- Processed: 733'833 objects in 99'771 frames (71/100 sequences reannotated)
- 25 FPS, 960x540 px
- Many camera angles, although some are repeated
- Classes originally car, bus, van, others (truck) - it was therefore reannotated so that two-wheelers and unknown vehicles are annotated, too
  - Not all instances were annotated, some were masked due to time constraints
- Masks are fixed (they were very bad)


### MIO-TCD

- https://tcd.miovision.com/challenge/dataset.html
- Huge, low-quality cam, surveillance type, many points of view
- Processed: 344'421 instances on 110'000 frames (only the test subset of the obejct localization part is annotated)
- Pedestrian + 9 vehicle classes (+ background)


### AAU RainSnow

- https://www.kaggle.com/datasets/aalborguniversity/aau-rainsnow
- Small, rain, snow, night, ... Surveillance-type, several points of view
- Processed: 10'545 instances on 1'899 frames
- Video, not sure about the FPS
- Classes: pedestrian, bicycle, car, motorbike, bus, truck
- Only RGB part is used
- We ignore this folders (annotated vehicles in them are impossible to detect because of lens flare): `["Egensevej-1", "Egensevej-3", "Egensevej-5"]`
- Transporters are of class "truck", the same as all trucks...
- Several errors in the dataset! See https://www.kaggle.com/datasets/aalborguniversity/aau-rainsnow/discussion/367019


### MTID

- https://www.kaggle.com/datasets/andreasmoegelmose/multiview-traffic-intersection-dataset?select=infrastructure-mscoco.json
- Small, two cams (one high quality, one less so), surveillance type
- Processed: 64'979 objects on 5'399 frames
- Needs masking! Both footages contain parked vehicles which are not annotated. Masks have to be applied manually
- 30 FPS
- Classes: Bicycle, Car, Bus, Lorry (transporter is annotated as a truck - lorry)
- Unannotated images from the drone footage!: `[1, 31], [659, 659], [1001, 1318], [3301, 3327]`
  - See https://www.kaggle.com/datasets/andreasmoegelmose/multiview-traffic-intersection-dataset/discussion/367060

### NDIS Park

- https://zenodo.org/record/6560823
- Tiny, HQ cam, surveillance type, many vehicles in one image
- Processed: 3'302 objects on 142 frames
- Only train and validation subsets are used
- Everything is annotated by a single class - `ndis.py` reannotates it because it contains transporters and trailers


### VisDrone
- https://github.com/VisDrone/VisDrone-Dataset
- Small, HQ cam, surveillance-type (drone)
- Processed: 47'720 objects on 1'610 frames
- Classes: pedestrian, person, car, van, bus, truck, motor, bicycle, awning-tricycle, tricycle
  - https://github.com/VisDrone/VisDrone2018-DET-toolkit
- Trailers are `unknown` class


- Pipeline for processing the original downloaded datasets (DOES NOT APPLY FOR DETRAC, because it was reannotated):
```
export d="aau" 
py $d.py
py apply_masks.py $d
py split_dataset_into_subsets.py $d
```
  - With visualizations:
```
export d="aau"
py $d.py
py apply_masks.py $d
py split_dataset_into_subsets.py $d
py visualize_dataset.py $d
py visualized_to_video.py $d
```
  - DETRAC only needs applying masks and splitting dataset into subsets, if already downloaded from LabelBox


# Models FLOPS and params

```
yolov8_f_352x192.py:                          0.0357866 GFLOPS, 0.0590577 M params
yolov8_f_384x224.py:                          0.0455466 GFLOPS, 0.07516444 M params
yolov8_f_448x256.py:                          0.0607288 GFLOPS, 0.100219 M params
yolov8_f_512x288.py:                          0.07808 GFLOPS,   0.1288533 M params
yolov8_l_mobilenet_v2_512x288_indices_246.py: 0.56 GFLOPS,      2.1333 M params
yolov8_m_640x384.py:                          7.85 GFLOPS,      18.3907 M params
yolov8_n_448x256.py:                          0.35408 GFLOPS,   0.78815 M params
yolov8_n_512x288.py:                          0.45525 GFLOPS,   1.01333 M params
yolov8_n_640x384.py:                          0.75875 GFLOPS,   1.68888 M params
yolov8_p_384x224.py:                          0.0881066 GFLOPS, 0.1789511 M params
yolov8_p_448x256.py:                          0.117475 GFLOPS,  0.2386 M params
yolov8_p_512x288.py:                          0.15104 GFLOPS,   0.30677 M params
yolov8_s_640x384.py:                          2.63822 GFLOPS,   3.73813 M params
```


# Testing

Test results can be found in `tests/test_results.json`.

All models (all input resolutions, both static and dynamic shapes) were tested
on all devices. Except when a model did not fit into memory, all batch sizes
were used. ONNX Runtime was used for tests on all devices, but on NVIDIA Jetson,
TensorRT was used as well. Quantization was only used with TensorRT.


### Test results format:
```py
data = {
    "device": { # Eg. "nano", "agx", "rpi", ...
        "model": { # Eg. "yolov8_f"
            "input_shape": { # Eg. "352x192"
                "backend": { # "tensorrt" or "onnxruntime"
                    "quantization": { # "none", "fp16", "int8"
                        "model_shape": { # "dynamic" or "static"
                            "batch_size": { # Eg. "1", "8", "32"...
                                "bbox_mAP": "float number in a string",
                                "bbox_mAP_50": "float number in a string",
                                "bbox_mAP_75": "float number in a string",
                                "bbox_mAP_s": "float number in a string",
                                "bbox_mAP_m": "float number in a string",
                                "bbox_mAP_l": "float number in a string",
                                "fps": "float number in a strin"g
                            }}}}}}}}

```

### Run ONNX Runtime on a single thread

- Edit `mmdeploy/mmdeploy/backend/onnxruntime/wrapper.py` and add before a line with `InferenceSession` creation:
```
session_options.intra_op_num_threads = 1
session_options.inter_op_num_threads = 1
```

### Get precision and recall values from the test script (and plot PR curve)

- To get precision and recall values after a test of a deployed model (to be saved to a file `coco_eval.json`):
  - Edit `mmdetection/mmdet/evaluation/metrics/coco_metric.py` - in `compute_metrics()`, after `coco_eval.summarize()` under `else`:
```
# save the precisions, iou, recall, cls and so on to a file
import json
import os
path = os.environ.get('RESULTS_FILEPATH')
if path is not None:
    results = {
        'precision': coco_eval.eval['precision'].tolist(),
        'recall': coco_eval.eval['recall'].tolist(),
        'scores': coco_eval.eval['scores'].tolist(),
        'iouThrs': coco_eval.eval["params"].iouThrs.tolist(),
        'recThrs':coco_eval.eval["params"].recThrs.tolist(),
        'catIds': [int(i) for i in coco_eval.eval["params"].catIds],
        'maxDets': [int(i) for i in coco_eval.eval["params"].maxDets],
        'areaRng': [[int(i) for i in l] for l in coco_eval.eval["params"].areaRng],
        'areaRngLbl': coco_eval.eval["params"].areaRngLbl,
        }
    with open(path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {path}")
else:
    print("RESULTS_FILEPATH not set, so results not saved")
```
  - Before running, `export RESULTS_FILEPATH=.../coco_eval.json`
    - Preferably, put it into a work dir, as that's where our scripts expect it
  - Run test example:
    - `for d in ./working_dir_yolov8_*; do export RESULTS_FILEPATH=/home/tedro/Desktop/proj/$d/coco_eval.json; python3 test_deployed.py deploy/config_onnxruntime_dynamic_batch.py $d onnxruntime_dynamic_batch.onnx cuda -b 8; done`
  - Example of then plotting a PR curve:
    - `python3 pr_curve.py working_dir_yolov8_f_352x192/ working_dir_yolov8_f_384x224/ working_dir_yolov8_f_448x256/ working_dir_yolov8_f_512x288/`


# Requirements

Provided in `requirements.txt`, however:
- Installing OpenMMLab libraries by cloning and `pip install -e .` is recommended, because many of their files need to be modified... MMYOLO and MMDeploy needs to be installed this way because we use their tools (from their repository).
- Newer onnxruntime should be okay. Be sure to use the GPU module version for inference using CUDA (onnxruntime-gpu)
- To use Python 3.6, some libraries (protobuf, MMYOLO, MMCV) need editing their `setup.py` script to allow for py3.6. Not a great solution but no compatibility issues were encountered


# FAQ and Frequent Errors

- Yes, when training, deploying and so on, MM libraries output a million warnings
- `IndexError: tensors used as indices must be long, byte or bool tensors`
  - A problem with MMDetection, occured on Raspberry Pi
  - Fix: In `/home/pi/bp/mmdetection/mmdet/structures/bbox/base_boxes.py`, change a line containing `bboxes = bboxes[index]` to `boxes = boxes[index.long()]`
    - https://stackoverflow.com/questions/60873477/indexerror-tensors-used-as-indices-must-be-long-byte-or-bool-tensors
- `operation.cpp:203:DCHECK(!i->is_use_only()) failed.` on Jetsons:
  - Solved by using older JP: 4.6.3 (older TensorRT version might work too)
  - Doesn't seem to affect static models though
  - The problem is in tensorrt pip package, `libnvinfer.so.8` file
  - TensorRT 8.5.2.2 didn't work, 8.2.1.8 and 8.2.1.9 worked (JP5.1.1 vs JP4.6.3)
- One of our scripts isn't working? Try running it from the root repo path or from the dir it is in :/
- `TensorRT is not available, please install TensorRT and build TensorRT custom ops first.`
  - Probably needs installing tensorrt libs: `sudo apt install -y tensorrt tensorrt-dev tensorrt-libs nvidia-tensorrt nvidia-tensorrt-dev`
- `module 'torch.distributed' has no attribute 'ReduceOp'`
  - You downloaded a bad version of PyTorch. Follow this README for working PyTorch downloads
- `AttributeError: 'MMLogger' object has no attribute '_cache'`
  - MMengine bug. Explained in this README (search for `_cache`)
- `mmdetection/mmdet/datasets/coco.py` might need editing (I don't remember the error message):
  - Instead of: `self.cat_ids = self.coco.get_cat_ids(cat_names=...`
  - It needs: `self.cat_ids = self.coco.get_cat_ids()`
- Windows: `io.UnsupportedOperation "fileno"`
  - In MMEngine's file `collect_env.py`, line 117 needs `__stdout__` instead of `stdout`
- `list index out of range` after validation?
  - `mmdet/evaluation/metrics/coco_metric.py` needs editing around line 216 for some reason:
  - Before line 212, add `if len(self.cat_ids) > label:` (and indent the next code block)
- `pad_param is not found in results` - a problem with the transformation pipeline in config. Shouldn't occur with our config files
- `visualize_pipeline.py` is too slow? That's because MMDet's `tools/misc/browse_dataset.py` is too slow. Fix:
  - In `mmengine/structures/base_data_element.py`, change `meta = copy.deepcopy(metainfo)` to `meta = metainfo`
    - Not guarranteed! Be careful with this
- Problem with unused parameters when training?:
  - https://github.com/pytorch/pytorch/issues/43259
  - https://github.com/open-mmlab/mmdetection/issues/7298
  - Put this in the model config: `find_unused_parameters = True`
- Problem with MMDeploy when deploying or testing? Make sure you built the library right
  - It probably needs custom ops for a specific backend (onnx or tensorrt or whatever)
  - https://github.com/open-mmlab/mmdeploy/issues/439#issuecomment-1120123144
  - Most often, the backend needs to be installed from source and then, MMDeploy needs to be built with the custom ops, for example:
    - `cmake .. -DMMDEPLOY_TARGET_BACKENDS="ort" -DONNXRUNTIME_DIR=/home/pi/bp/onnxruntime/build/Linux/Release/`
    - Make sure to update LD_LIBRARY_PATH or PATH env variables if it doesn't work by itself


# Installing on NVIDIA Jetson

Older Jetpack 4.6.3 needs to be used, otherwise (when using 5.1.1) the TensorRT
will fail to deploy some of the models. The error is unexplainable:
`operation.cpp:203:DCHECK(!i->is_use_only()) failed.`. Seems like a problem with
newer TensorRT version as the error originates from `libnvinfer`.

Keep in mind that quantization to INT8 doesn't work on Nano (or might work but
it's not optimized for it)


### Personal installation pipeline:

- https://github.com/open-mmlab/mmdeploy/blob/master/docs/en/01-how-to-build/jetsons.md
    - Conda is not needed; inference SDK is not needed

- Basic: ```
    sudo apt update && sudo apt -y upgrade && sudo apt -y autoremove
    sudo apt install -y python3-pip python3-dev python3-setuptools python3-wheel libpython3-dev
    if ! [ -f /usr/bin/pip ]; then sudo ln -s /usr/bin/pip3 /usr/bin/pip; fi
    ```
- `sudo apt install -y protobuf-compiler libprotobuf-dev libssl-dev libjpeg-dev zlib1g-dev libavcodec-dev libavformat-dev libswscale-dev libopenblas-base libopenmpi-dev libopenblas-dev vim tree htop tcpdump ncdu pkg-config libhdf5-dev libspdlog-dev build-essential software-properties-common cython3 tensorrt tensorrt-dev tensorrt-libs nvidia-tensorrt nvidia-tensorrt-dev libgeos-dev tmux`
    - ! `tensorrt-dev` `tensorrt-libs` and `nvidia-tensorrt-dev` are not available on every device. They are probably only needed on Xavier NX with JP 5.1.1. So if you get an error, remove these from the command
- `pip install --upgrade pip`
- JP 5.1.1:
    - `pip install numpy==1.21.0 protobuf==3.20.2 matplotlib==3.7.1`
- JP 4.6.1:
```
pip install numpy==1.19.4
pip install matplotlib==3.3.4
# Protobuf
if ! [ -d ~/protobuf ]; then git clone --recursive https://github.com/protocolbuffers/protobuf ~/protobuf; fi
cd ~/protobuf && git checkout tags/v3.20.2 && git submodule update --init --recursive
cd ~/protobuf && ./autogen.sh && ./configure && make -j`nproc`
sudo make install
sed -i "s/>=3.7/>=3.6/g" ~/protobuf/python/setup.py
pip install -e ~/protobuf/python/
```
    - Matplotlib might not work on NX and AGX with JP 4.6.3
- `pip install pycuda aiohttp scipy versioned-hdf5 packaging opencv-python Pillow tqdm`
    - On JP 4.6.3 PyCuda won't install
        - It couldn't find `xlocale.h`
            - `sudo ln -s /usr/include/locale.h /usr/include/xlocale.h` helps
- `sudo pip3 install -U jetson-stats`
- GitHub repos:
```
if ! [ -d ~/vision ]; then git clone --recursive https://github.com/pytorch/vision ~/vision; fi
if ! [ -d ~/mmcv ]; then git clone --recursive https://github.com/open-mmlab/mmcv ~/mmcv; fi
if ! [ -d ~/mmdeploy ]; then git clone --recursive https://github.com/open-mmlab/mmdeploy ~/mmdeploy; fi
if ! [ -d ~/mmdetection ]; then git clone --recursive https://github.com/open-mmlab/mmdetection ~/mmdetection; fi
if ! [ -d ~/mmengine ]; then git clone --recursive https://github.com/open-mmlab/mmengine ~/mmengine; fi
if ! [ -d ~/mmyolo ]; then git clone --recursive https://github.com/open-mmlab/mmyolo ~/mmyolo; fi
cd ~/vision; git checkout tags/v0.11.1
cd ~/mmcv; git checkout tags/v2.0.0
cd ~/mmdeploy; git checkout tags/v1.0.0
cd ~/mmdetection; git checkout tags/v3.0.0
cd ~/mmengine; git checkout tags/v0.7.2
cd ~/mmyolo; git checkout tags/v0.4.0
```
- Update .bashrc:
```
echo 'export PATH="$PATH:/usr/local/cuda/bin"' >> ~/.bashrc
echo 'export PATH="$PATH:$HOME/.local/bin"' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH="/usr/local/cuda/lib64:$LD_LIBRARY_PATH"' >> ~/.bashrc
echo 'export OPENBLAS_CORETYPE=ARMV8' >> ~/.bashrc
echo 'export CUDACXX="/usr/local/cuda/bin/nvcc"' >> ~/.bashrc
echo 'export TENSORRT_DIR=/usr/include/aarch64-linux-gnu' >> ~/.bashrc
echo 'export PPLCV_DIR=~/ppl.cv/' >> ~/.bashrc
```
- CMake:
```
sudo apt-get purge cmake -y
export CMAKE_VER=3.23.1
export ARCH=aarch64
wget https://github.com/Kitware/CMake/releases/download/v${CMAKE_VER}/cmake-${CMAKE_VER}-linux-${ARCH}.sh
chmod +x cmake-${CMAKE_VER}-linux-${ARCH}.sh
sudo ./cmake-${CMAKE_VER}-linux-${ARCH}.sh --prefix=/usr --skip-license
cmake --version
```
- JP 5.1.1
    - `wget https://nvidia.box.com/shared/static/ssf2v7pf5i245fk4i0q926hy4imzs2ph.whl -O ~/torch-1.11.0-cp38-cp38-linux_aarch64.whl`
    - `pip install ~/torch-1.11.0-cp38-cp38-linux_aarch64.whl`
- JP 4.6.1
    - `wget https://nvidia.box.com/shared/static/fjtbno0vpo676a25cgvuqc1wty0fkkg6.whl -O ~/torch-1.10.0-cp36-cp36m-linux_aarch64.whl`
    - `pip install ~/torch-1.10.0-cp36-cp36m-linux_aarch64.whl`
- `pip install -e ~/vision`
- ppl.cv:
```
if ! [ -d $PPLCV_DIR ]; then git clone https://github.com/openppl-public/ppl.cv.git $PPLCV_DIR; fi
cd $PPLCV_DIR
./build.sh cuda
```
- Nano MMLabs: ignore requirements: 
```
sed -i "s/>=3.7/>=3.6/g" ~/mmcv/setup.py
sed -i "s/>=3.7/>=3.6/g" ~/mmengine/setup.py
```
- JP 4.6.3 MMLabs: comment out lines 290+291 in:
    - `~/mmengine/mmengine/logging/logger.py`
    - Lines that should be commented out:
```
for logger in MMLogger._instance_dict.values():
    logger._cache.clear()
```
- MMLabs:
```
MMCV_WITH_OPS=1 pip install -v -e ~/mmcv/
pip install -e ~/mmengine/
pip install -e ~/mmdetection/
pip install -e ~/mmyolo/
cd ~/mmdeploy
mkdir -p build && cd build
cmake .. -DMMDEPLOY_TARGET_BACKENDS="trt"
make -j`nproc` && make install
pip install -e ~/mmdeploy/
```
- ONNXRuntime with `CUDAExecutionProvider`:
```
pip uninstall -y onnxruntime
git clone --recursive https://github.com/microsoft/onnxruntime ~/onnxruntime
cd ~/onnxruntime/
git checkout tags/v1.11.0
./build.sh --config Release --update --build --parallel --build_wheel --use_cuda --cuda_home /usr/local/cuda --cudnn_home /usr/lib/aarch64-linux-gnu
pip install ~/onnxruntime/build/Linux/Release/dist/onnxruntime_gpu-*-linux_aarch64.whl
```
- If we don't want to create a new `calib_file` for every deployed model (but instead, reuse a calibration file for each model with the same input resolution), edit `~/mmdeploy/mmdeploy/utils/config_utils.py:340` from:
```
create_calib = calib_config.get('create_calib', False)
if create_calib:
    calib_filename = calib_config.get('calib_file', 'calib_file.h5')
    return calib_filename
else:
    return None
```
to:
```
return calib_config.get('calib_file', 'calib_file.h5')
```
- Additionally, matplotlib version needs to be checked, because some package might have reinstalled it to an older version (2.1.1 version is old)
- DONE


### Additional Info

- Packages used and working on Jetsons (src means building/installing from source):
```
Package             Nano           Xavier NX      AGX XAVIER
mmcv                2.0.0 src      2.0.0 src      2.0.0 src
mmdeploy            1.0.0 src      1.0.0 src      1.0.0 src
mmdet               3.0.0 src      3.0.0 src      3.0.0 src
mmengine            0.7.2 src      0.7.2 src      0.7.2 src
mmyolo              0.4.0 src      0.4.0 src      0.4.0 src
numpy               1.19.4         1.21.0         1.21.0
onnx                1.13.1         1.13.1         1.13.1
onnruntime_gpu      1.11.0         1.11.0         1.11.0
protobuf            3.20.2 src     3.20.2         3.20.2 src
pycuda              2022.1         2022.2.2       2022.2.2
scipy               0.19.1         1.3.3          1.3.3
torch               1.10.0         1.11.0         1.11.0
torchvision         0.11.0 src     0.11.0 src     none
```

- For ONNX Runtime inference on Jetsons (to utilize GPU), the GPU version needs to be built from source:
  - https://onnxruntime.ai/docs/build/eps.html#nvidia-jetson-tx1tx2nanoxavier
  - Be sure to uninstall `onnxruntime` if present
  - It might need `libpython3.8-dev` or `libpython3.6-dev`

- PyTorch install on Jetsons...
  - https://forums.developer.nvidia.com/t/pytorch-for-jetson/72048
    - Other sources just don't work...
    - Nano: v1.10.1, NX and AGX (if JP 5.1.1 is used, I guess) v1.11.0

- OpenMMLabs on Nano? (JP 4.6.3)
  - Needs protobuf 3.20.2 from source, I guess
    - If it won't work with python3.6, `setup.py` needs to be edited manually to accept the old python
  - We can then install MMDeploy 1.0.0
    - This installs MMEngine 0.4.0 - we will upgrade this later
  - MMCV 2.0.0 (or maybe higher)
    - Needs python>=3.7, so we need to modify `setup.py` to work with 3.6...
    - Don't forget to build with `MMCV_WITH_OPS`
  - MMEngine needs to be 0.7.2 (or maybe higher)
    - Again, needs python>=3.7, so `setup.py` needs to be edited
  - Lines 290 and 192 in `mmengine/logging/logger.py` need to be commented out. Otherwise it throws `AttributeError: 'MMLogger' object has no attribute '_cache'`
  - Then, libgeos needs to be installed: `sudo apt-get install libgeos-dev`
    - Otherwise: `OSError: Could not find lib geos_c or load any of its variants ['libgeos_c.so.1', 'libgeos_c.so'].`
  - Finally, install newer matplotlib (3.3.4) if an older version was installed
