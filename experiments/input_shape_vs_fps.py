import os
import json
from pprint import pprint

tests_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tests")
with open(os.path.join(tests_path, "test_results.json")) as f:
    data_in = json.load(f)


data = {}
# for device in ["nano", "nx", "agx"]:
# for device in ["nx", "agx"]:
# for device in ["agx"]:
for device in ["nano"]:
    for model in data_in[device]:
        for input_resolution in data_in[device][model]:
            # ort_fps = data_in[device][model][input_resolution]["onnxruntime"]["none"]["dynamic"]["16"]["fps"]
            trt_fps = data_in[device][model][input_resolution]["tensorrt"]["none"]["dynamic"]["16"]["fps"]

            model_name = model.replace("yolov8_", "YOLOv8-")
            if model_name.endswith("m"):
                model_name += "edium"
            elif model_name.endswith("s"):
                model_name += "mall"
            elif model_name.endswith("f"):
                model_name += "emto"
            elif model_name.endswith("p"):
                model_name += "ico"
            elif model_name.endswith("n"):
                model_name += "ano"
            if "mobilenet" in model_name:
                model_name = "MobileNetV2"
            model_name = model_name + " " + input_resolution

            if model_name not in data:
                data[model_name] = {}
            if input_resolution not in data[model_name]:
                data[model_name][input_resolution] = {}

            data[model_name][input_resolution] = float(trt_fps)

pprint(data)


import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({"font.size": 10})
plt.rcParams.update({"font.family": "serif"})
plt.rcParams.update({"font.serif": "cmr10"})
plt.rcParams.update({"text.usetex": True})

organized_data = {}
for key, fps_data in data.items():
    model_name, input_resolution = key.rsplit(" ", 1)
    if model_name not in organized_data:
        organized_data[model_name] = {}
    organized_data[model_name][input_resolution] = fps_data[input_resolution]

filtered_data = {k: v for k, v in organized_data.items() if len(v) >= 2}

n_models = len(filtered_data)
n_rows = 3
n_cols = int(np.ceil(n_models / n_rows))

fig, axs = plt.subplots(n_rows, n_cols, figsize=(6, 4.4), sharex=False, gridspec_kw={'height_ratios': [3, 3, 4]})
axs = axs.flatten()


fig.suptitle("Effect of Model's Input Resolution on Inference Speed", fontsize=14)

model_names = ["YOLOv8-nano", "YOLOv8-pico", "YOLOv8-femto"]
for i, model_name in enumerate(model_names):
    input_resolutions = list(filtered_data[model_name].keys())

    # Find the highest available input resolution for the model
    base_resolution = max(input_resolutions, key=lambda s: int(s.split("x")[0]) * int(s.split("x")[1]))
    base_fps = filtered_data[model_name][base_resolution]

    actual_fps = []
    expected_fps = []
    for resolution in input_resolutions:
        actual_fps.append(filtered_data[model_name][resolution])
        expected_fps.append(base_fps * (int(base_resolution.split("x")[0]) * int(base_resolution.split("x")[1])) / (int(resolution.split("x")[0]) * int(resolution.split("x")[1])))

    # positions for the bars
    bar_width = 0.30
    positions_actual = np.arange(len(input_resolutions))
    positions_expected = [p + bar_width for p in positions_actual]

    axs[i].barh(positions_expected, expected_fps, height=bar_width, label="Expected FPS")
    axs[i].barh(positions_actual, actual_fps, height=bar_width, label="Actual FPS")

    axs[i].set_xlim(0, max(expected_fps) * 1.5)

    axs[i].set_title(f"{model_name}", fontsize=12)
    axs[i].set_yticks([p + bar_width / 2 for p in positions_actual])
    axs[i].set_yticklabels([f"${resolution.split('x')[0]} \\times {resolution.split('x')[1]}$" for resolution in input_resolutions])
    axs[i].set_ylabel("Input Resolution")
    axs[i].set_xlabel("FPS")

    legend = plt.legend(loc='center left', bbox_to_anchor=(1, 2.5))
    # axs[i].legend()
    axs[i].legend(loc='upper right', borderaxespad=0.25)

for j in range(i + 1, n_rows * n_cols):
    fig.delaxes(axs[j])
plt.tight_layout()

plt.savefig("/home/tedro/Desktop/d_projekty/bp/proj/doc/figures/input_resolution_vs_fps.pdf", bbox_inches='tight', transparent=True)
print("DONE")
