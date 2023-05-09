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

def get_max_input_shape(model_data):
    available_shapes = list(model_data.keys())
    max_shape = max(available_shapes, key=lambda shape: int(shape.split('x')[0]) * int(shape.split('x')[1]))
    return max_shape

def get_device_name(device):
    device_names = {
        "nano": "NVIDIA Jetson Nano",
        "agx": "NVIDIA Jetson AGX",
        "nx": "NVIDIA Jetson NX",
        "rpi": "Raspberry PI",
        "mx150": "NVIDIA MX150",
        "intel_i7": "Intel Core i7-9850H"
    }
    return device_names.get(device, device)

def create_subplot(device, backend, ax):
    model_names = {
        "yolov8_f": "femto",
        "yolov8_p": "pico",
        # "yolov8_n": "nano",
        # "yolov8_s": "small",
        "yolov8_m": "medium",
        "yolov8_l_mobilenet_v2": "MobileNetV2"
    }
    models = list(model_names.keys())
    fps_values = []

    max_shapes = {}
    for model in models:
        max_shape = get_max_input_shape(data[device][model])
        static_fps_value = float(data[device][model][max_shape][backend]['none']['static']['1']['fps'])
        dynamic_fps_value = float(data[device][model][max_shape][backend]['none']['dynamic']['1']['fps'])
        max_shapes[model] = max_shape

        fps_values.append((model, static_fps_value, dynamic_fps_value))

    fps_values.sort(key=lambda x: x[1], reverse=True)

    # labels = [f"{model_names[model]}\n{max_shape.replace('x', ' $\\times$ ')}" for model, _, _ in fps_values]
    labels = [f"{model_names[model]}\n${max_shapes[model].split('x')[0]}" + r"\times" + f"{max_shapes[model].split('x')[1]}$" for model, _, _ in fps_values]
    static_fps = [static_fps for _, static_fps, _ in fps_values]
    dynamic_fps = [dynamic_fps for _, _, dynamic_fps in fps_values]

    ind = range(len(models))
    width = 0.35

    ax.barh([i + width for i in ind], static_fps, width, label='Static Shape')
    ax.barh(ind, dynamic_fps, width, label='Dynamic Shape')

    ax.set_title(f'{get_device_name(device)}\,--\,{backend.capitalize().replace("Onnxruntime", "ONNX Runtime").replace("Tensorrt", "TensorRT")}', fontsize=12)
    ax.set_yticks([i + width / 2 for i in ind])
    ax.set_yticklabels(labels)
    ax.set_xlabel('FPS')
    ax.set_ylabel('Model')

    ax.legend()

    # ax.xaxis.set_major_locator(plt.MaxNLocator(5))  # Limit number of ticks on x-axis

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 4.25))
create_subplot('rpi', 'onnxruntime', ax1)
create_subplot('nano', 'tensorrt', ax2)
plt.tight_layout()

fig.suptitle("Inference Speeds: Static vs. Dynamic YOLOv8 Models", y=1.02, fontsize=14)
figures_path = "/home/tedro/Desktop/d_projekty/bp/proj/doc/figures/"
figure_name = "static_dynamic_comparison.pdf"
fig.savefig(os.path.join(figures_path, figure_name), bbox_inches='tight', transparent=True)
