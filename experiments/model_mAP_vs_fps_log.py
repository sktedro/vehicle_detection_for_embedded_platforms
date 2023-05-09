import os
import json
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({"font.size": 10})
plt.rcParams.update({"font.family": "serif"})
plt.rcParams.update({"font.serif": "cmr10"})
plt.rcParams.update({"text.usetex": True})

tests_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tests")
with open(os.path.join(tests_path, "test_results.json")) as f:
    data = json.load(f)

with open(os.path.join(tests_path, "mAP", "test_all", "test_all_results.json")) as f:
    mAP_data = json.load(f)
    mAP_data = mAP_data[list(mAP_data.keys())[0]]

model_names = {
    "yolov8_m": "medium",
    "yolov8_s": "small",
    "yolov8_l_mobilenet_v2": "MobileNetV2",
    "yolov8_n": "nano",
    "yolov8_p": "pico",
    "yolov8_f": "femto",
}
device_names = {
    "nano": "NVIDIA Jetson Nano",
    "agx": "NVIDIA Jetson AGX Xavier",
    "nx": "NVIDIA Jetson Xavier NX",
    "rpi": "Raspberry PI",
    "mx150": "NVIDIA MX150",
    "intel_i7": "Intel Core i7-9850H",
}

def get_mAP(device, model, resolution, backend, quantization, model_shape, batch_size):
    try:
        mAP = float(data[device][model][resolution][backend][quantization][model_shape][batch_size]["bbox_mAP"])
    except Exception as e:
        print("==================================================")
        print("WARNING: Exception ignored:")
        from traceback import print_exc
        print_exc()
        print("Fallback to tests/mAP/test_all/test_all_results.json")
        backends = mAP_data[model][resolution]
        key = list(backends.keys())[0]
        batch_sizes = backends[key][quantization][model_shape]
        key = list(batch_sizes.keys())[0]
        results = batch_sizes[key]
        mAP = float(results["bbox_mAP"])
        print("==================================================")
    return mAP

def f(device, backend, batch_size):
    model_shape = "dynamic"

    input_resolutions_lists = [list(data[device][model].keys()) for model in data[device]]
    input_resolutions = []
    for l in input_resolutions_lists:
        input_resolutions += l
    input_resolutions = list(set(input_resolutions))
    input_resolutions.sort(reverse=True)

    markers = ['o', 's', 'P', 'H', 'X', 'v']
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']

    if device == "nx":
        fig, ax = plt.subplots(figsize=(6, 4.5))
    else:
        fig, ax = plt.subplots(figsize=(6, 3))

    quantizations = set()
    for i, model in enumerate(model_names.keys()):
        for j, resolution in enumerate(input_resolutions):
            if resolution not in data[device][model]:
                continue
            for k, quantization in enumerate(data[device][model][resolution][backend]):
                mAP = get_mAP(device, model, resolution, backend, quantization, model_shape, batch_size)
                fps = float(data[device][model][resolution][backend][quantization][model_shape][batch_size]["fps"])

                # Normalize mAP to percentage
                mAP *= 100

                size = 50
                linewidth = 1.25
                
                if quantization == "none":
                    fillstyle = 'full'
                    edgecolor = 'k'
                    if backend == "onnxruntime":
                        edgecolor = "none"
                    ax.scatter(fps, mAP, marker=markers[i], s=size, c=colors[j], edgecolor=edgecolor, facecolors=fillstyle, linewidth=linewidth)

                elif quantization == "fp16":
                    fillstyle = 'full'
                    edgecolor = colors[j]
                    ax.scatter(fps, mAP, marker=markers[i], s=size, c=colors[j], edgecolor=edgecolor, facecolors=fillstyle, linewidth=linewidth)

                elif quantization == "int8":
                    fillstyle = 'None'
                    edgecolor = colors[j]
                    ax.scatter(fps, mAP, marker=markers[i], s=size, edgecolor=edgecolor, facecolors=fillstyle, linewidth=linewidth)

                if quantization == "none":
                    quantizations.add("FP32")
                else:
                    quantizations.add(quantization)

    for collection in ax.collections:
        collection.set_zorder(3)

    ax.grid(True, linestyle="--", alpha=0.4)

    if backend == "tensorrt":
        for i, model in enumerate(model_names.keys()):
            for j, resolution in enumerate(input_resolutions):
                if resolution not in data[device][model]:
                    continue
                fps_list = []
                mAP_list = []
                for k, quantization in enumerate(data[device][model][resolution][backend]):
                    fps = float(data[device][model][resolution][backend][quantization][model_shape][batch_size]["fps"])
                    # mAP = float(data[device][model][resolution][backend][quantization][model_shape][batch_size]["bbox_mAP"])
                    mAP = get_mAP(device, model, resolution, backend, quantization, model_shape, batch_size)
                    fps_list.append(fps)
                    mAP_list.append(mAP)
                mAP_list = [v * 100 for v in mAP_list]

                ax.plot(fps_list, mAP_list, color=colors[j], linestyle='-', linewidth=1, alpha=0.15)

    else:
        for i, model in enumerate(model_names.keys()):
            fps_list = []
            mAP_list = []
            for j, resolution in enumerate(input_resolutions):
                if resolution not in data[device][model]:
                    continue
                fps = float(data[device][model][resolution][backend]["none"][model_shape][batch_size]["fps"])
                # mAP = float(data[device][model][resolution][backend][quantization][model_shape][batch_size]["bbox_mAP"])
                mAP = get_mAP(device, model, resolution, backend, "none", model_shape, batch_size)
                fps_list.append(fps)
                mAP_list.append(mAP)
            mAP_list = [v * 100 for v in mAP_list]

            ax.plot(fps_list, mAP_list, color="k", linestyle='-', linewidth=1, alpha=0.15)


    # legend for model shapes
    handles1, labels1 = [], []
    for i, model_name in enumerate(model_names.values()):
        handles1.append(ax.scatter([], [], marker=markers[i], s=50, color="white", edgecolor="black", label=model_name))
        labels1.append(model_name)

    # legend for input resolutions
    handles2, labels2 = [], []
    for j, resolution in enumerate(reversed(list(np.sort(input_resolutions)))):
        resolution = f"${resolution.split('x')[0]}" + r"\times" + f"{resolution.split('x')[1]}$"
        handles2.append(ax.scatter([], [], marker="o", s=50, color=colors[j], label=resolution))
        labels2.append(resolution)

    # legend for quantization levels
    if backend == "tensorrt":
        handles3, labels3 = [], []
        handles3.append(ax.scatter([], [], marker="o", s=50, c="blue", facecolors="full", edgecolor="black", label="FP32"))
        handles3.append(ax.scatter([], [], marker="o", s=50, c="blue", facecolors="full",                    label="FP16"))
        handles3.append(ax.scatter([], [], marker="o", s=50,           facecolors="none", edgecolor="blue",  label="INT8"))
        labels3 = [q for q in ["FP32", "FP16", "INT8"] if q in [s.upper() for s in quantizations]]
        # labels3 = [s.upper() for s in quantizations]

    legend1 = ax.legend(handles1, labels1, title="Models", loc="lower left", bbox_to_anchor=(0, 0))
    legend2 = ax.legend(handles2, labels2, title="Input resolutions", loc="lower left", bbox_to_anchor=(0.29, 0))
    if backend == "tensorrt":
        legend3 = ax.legend(handles3, labels3, title="Precisions", loc="lower left", bbox_to_anchor=(0.545, 0))

    ax.add_artist(legend1)
    ax.add_artist(legend2)
    if backend == "tensorrt":
        ax.add_artist(legend3)

    if device == "nx":
        yticks = np.arange(0.15, 0.625, 0.05)
    else:
        yticks = np.arange(0.25, 0.625, 0.05)
    yticks *= 100
    ax.set_yticks(yticks)
    ax.set_ylim(yticks[0])

    # ax.set_xticks(np.arange(0, 1200, 100))

    ax.set_xscale('log')
    if device == "nx":
        xticks = [10, 20, 50, 100, 200, 500, 1000]
    else:
        xticks = [0.1, 0.2, 0.5, 1, 2, 5, 10, 20]
    ax.set_xlim(xticks[0])
    ax.set_xticks(xticks)
    ax.set_xticklabels([str(t) for t in xticks])

    ax.set_xlabel('FPS (logarithmic scale)')
    ax.set_ylabel('Mean Average Precision')

    # ax.set_title(f'{device_names[device]} - {backend.capitalize().replace("Onnxruntime", "ONNX Runtime").replace("Tensorrt", "TensorRT")}', fontsize=12)
    # fig.suptitle("YOLOv8 - FPS vs mAP Comparison", fontsize=14)

    if device == "nx":
        fig.suptitle("mAP vs. FPS Comparison of All YOLOv8 Models", fontsize=12, y=0.97)
    else:
        fig.suptitle("mAP vs. FPS Comparison of All YOLOv8 Models", fontsize=12, y=1.01)
    ax.set_title(f'{device_names[device]} -- {backend.capitalize().replace("Onnxruntime", "ONNX Runtime").replace("Tensorrt", "TensorRT")} -- Batch Size {batch_size}', fontsize=11)

    output_path = "/home/tedro/Desktop/d_projekty/bp/proj/doc/figures/"
    fig.savefig(os.path.join(output_path, f"fps_vs_map_comparison_log_{device}.pdf"), bbox_inches='tight', transparent=True)

f("nx", "tensorrt", "32")
f("rpi", "onnxruntime", "1")