import onnxruntime
import mmcv
import numpy as np
import cv2
from PIL import Image

import sys, os
repo_path = os.path.join(os.path.dirname(__file__), '..')
sys.path.append(repo_path)
import paths


model_filepath = os.path.join(paths.working_dirpath, paths.deploy_onnx_filename)
print(model_filepath)
session = onnxruntime.InferenceSession(model_filepath, providers=['CPUExecutionProvider'])

session.get_modelmeta()
first_input_name = session.get_inputs()[0].name
first_output_name = session.get_outputs()[0].name

video_filepath = os.path.join(paths.proj_path, "vid", "day_hq.mp4")
video = mmcv.VideoReader(video_filepath)

frame = video.read()
frame = mmcv.imresize(frame, (640, 384))

frame = frame.transpose(2, 0, 1)
frame = frame.astype(np.float32)
# frame /= 255.
frame = np.expand_dims(frame, axis=0)


# This is for my own deploy script:
bboxes, labels = session.run(None, {first_input_name: frame})
assert len(bboxes) == 1
assert len(labels) == 1

bboxes = np.array(bboxes[0])
labels = np.array(labels[0])
print(len(labels))

img = frame[0].transpose(1, 2, 0).copy()
# img = frame[0].copy()
img_h, img_w, _ = img.shape

for i in range(len(bboxes)):
    if bboxes[i][4] < 0.7:
        continue

    x1, y1, x2, y2 = bboxes[i][:4].astype(int)
    confidence = round(bboxes[i][4] * 100)
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 1)
    img = cv2.putText(img, str(confidence) + ": cls=" + str(labels[i]), (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1)


# This for mmyolo's easydeploy (export.py):
# out = session.run(None, {first_input_name: frame})

# num_preds, bboxes, confidences, labels = session.run(None, {first_input_name: frame})
# assert len(bboxes) == 1
# assert len(labels) == 1

# num_preds = np.array(num_preds[0])
# bboxes = np.array(bboxes[0])
# confidences = np.array(confidences[0])
# labels = np.array(labels[0])
# print(len(labels))

# img = frame[0].transpose(1, 2, 0).copy()
# # img = frame[0].copy()
# img_h, img_w, _ = img.shape

# for i in range(len(bboxes)):
#     # if bboxes[i][4] < 0.7:
#     if confidences[i] < 0.7:
#         continue

#     x1, y1, x2, y2 = bboxes[i][:4].astype(int)
#     # confidence = round(bboxes[i][4] * 100)
#     confidence = round(confidences[i] * 100)
#     cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 1)
#     img = cv2.putText(img, str(confidence) + ": cls=" + str(labels[i]), (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1)

cv2.imwrite("output_image.jpg", img)
