import os
import json
import matplotlib.pyplot as plt

plt.rcParams.update({"font.size": 10})
plt.rcParams.update({"font.family": "serif"})
plt.rcParams.update({"font.serif": "cmr10"})
plt.rcParams.update({"text.usetex": True})

tests_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tests")
with open(os.path.join(tests_path, "test_results.json")) as f:
    data = json.load(f)

def get_device_name(device):
    device_names = {
        "nano": "NVIDIA Jetson Nano",
        "agx": "NVIDIA Jetson AGX Xavier",
        "nx": "NVIDIA Jetson Xavier NX",
        "rpi": "Raspberry PI",
        "mx150": "NVIDIA MX150",
        "intel_i7": "Intel Core i7-9850H"
    }
    return device_names.get(device, device)

def create_subplot(device, backend, ax, is_first, xticks, models):
    # TODO change for different amount of models
    # model_names = {
    #     "yolov8_f": "femto",
    #     "yolov8_p": "pico",
    #     "yolov8_n": "nano",
    #     "yolov8_l_mobilenet_v2": "MobileNetV2",
    #     "yolov8_s": "small",
    #     "yolov8_m": "medium",
    # }
    model_names = {
        "yolov8_f": "femto",
        "yolov8_p": "pico",
        "yolov8_n": "nano",
        "yolov8_l_mobilenet_v2": "MobileNetV2",
        "yolov8_s": "small",
        "yolov8_m": "medium",
    }
    for name in model_names.copy():
        if name not in models:
            del model_names[name]
    # model_names = {
    #     "yolov8_f": "femto",
    #     # "yolov8_p": "pico",
    #     # "yolov8_n": "nano",
    #     "yolov8_l_mobilenet_v2": "MobileNetV2",
    #     # "yolov8_s": "small",
    #     "yolov8_m": "medium",
    # }

    model_shapes = {
        "yolov8_f": "352x192",
        "yolov8_p": "512x288",
        "yolov8_n": "640x384",
        "yolov8_s": "640x384",
        "yolov8_m": "640x384",
        "yolov8_l_mobilenet_v2": "512x288"
    }

    models = list(model_names.keys())
    fps_values = []

    for model in models:
        shape = model_shapes[model]
        batch_sizes = [1, 2, 8, 32]
        model_fps_values = []

        for batch_size in batch_sizes:
            try:
                fps_value = float(data[device][model][shape][backend]["none"]["dynamic"][str(batch_size)]["fps"])
            except KeyError:
                fps_value = None
            model_fps_values.append(fps_value)

        fps_values.append((model, model_fps_values))

    fps_values.sort(key=lambda x: min([fps for fps in x[1] if fps is not None]), reverse=True)

    labels = [f"{model_names[model]}\n${model_shapes[model].split('x')[0]}" + r"\times" + f"{model_shapes[model].split('x')[1]}$" for model, _ in fps_values]
    # labels = [f"{model_names[model]}\n"+ r"\footnotesize" + f"${model_shapes[model].split('x')[0]}" + r"\times" + f"{model_shapes[model].split('x')[1]}$" for model, _ in fps_values]

    ind = range(len(models))
    batch_width = 0.175

    for batch_idx, batch_size in enumerate(batch_sizes):
        batch_fps = [fps_list[batch_idx] for _, fps_list in fps_values if fps_list[batch_idx] is not None]
        valid_indices = [i for i, (_, fps_list) in enumerate(fps_values) if fps_list[batch_idx] is not None]
        ax.barh([i - batch_width * batch_idx + 3*batch_width for i in valid_indices], batch_fps, batch_width, label=f'Batch Size {batch_size}')

    ax.set_title(f'{get_device_name(device)}\,--\,{backend.capitalize().replace("Onnxruntime", "ONNX Runtime").replace("Tensorrt", "TensorRT")}', fontsize=12)
    ax.set_yticks([i + batch_width * (len(batch_sizes) - 1) / 2 for i in ind])
    # ax.set_yticklabels(labels, fontsize=10)
    ax.set_yticklabels(labels)
    ax.set_ylabel('Model')

    ax.set_xscale('log')
    ax.set_xticks(xticks)
    ax.set_xticklabels([str(t) for t in xticks])
    ax.set_xlabel('FPS (logarithmic scale)')

    legend = ax.legend()
    # frame = legend.get_frame()
    # frame.set_linewidth(0.1)

    # if is_first:
    #     legend = ax.legend(fontsize=10, frameon=True, loc="center left", bbox_to_anchor=(1, 0.5))
    #     frame = legend.get_frame()
    #     frame.set_linewidth(0.1)


# TODO change for different amount of models
# fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(6, 9))
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(6, 5.6), gridspec_kw={'height_ratios': [3, 3, 4]})
# fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(6, 6))

create_subplot('rpi', 'onnxruntime', ax1, True,
               [0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50, 100, 200],
               ["yolov8_m", "yolov8_l_mobilenet_v2", "yolov8_f"])

create_subplot('nano', 'tensorrt', ax2, False,
               [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000],
               ["yolov8_m", "yolov8_l_mobilenet_v2", "yolov8_f"])

create_subplot('agx', 'tensorrt', ax3, False,
               [10, 20, 50, 100, 200, 500, 1000],
               ["yolov8_m", "yolov8_s", "yolov8_l_mobilenet_v2", "yolov8_f"])
plt.tight_layout()

# fig.suptitle("Inference Speeds: YOLOv8 Models on Different Devices\nwith Different Batch Sizes", y=1.02, fontsize=14)
# fig.suptitle("Effect of Batch Size on Inference Speeds of YOLOv8 models", y=1.02, fontsize=14)
fig.suptitle("Effect of Batch Size on Inference Speeds of YOLOv8 Models", y=1.02, fontsize=14)

figures_path = "/home/tedro/Desktop/d_projekty/bp/proj/doc/figures/"
figure_name = "batch_size_comparison.pdf"
fig.savefig(os.path.join(figures_path, figure_name), bbox_inches='tight', transparent=True)
