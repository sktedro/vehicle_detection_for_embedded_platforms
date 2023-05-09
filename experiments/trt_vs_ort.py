import os
import json
import matplotlib.pyplot as plt

# Update font settings
plt.rcParams.update({"font.size": 10})
plt.rcParams.update({"font.family": "serif"})
plt.rcParams.update({"font.serif": "cmr10"})
plt.rcParams.update({"text.usetex": True})

# Read data
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
        "agx": "NVIDIA Jetson AGX Xavier",
        "nx": "NVIDIA Jetson Xavier NX",
        "rpi": "Raspberry PI",
        "mx150": "NVIDIA MX150",
        "intel_i7": "Intel Core i7-9850H"
    }
    return device_names.get(device, device)

def create_subplot(device, batch_size, ax):
    model_names = {
        "yolov8_f": "femto",
        "yolov8_p": "pico",
        "yolov8_n": "nano",
        "yolov8_s": "small",
        "yolov8_m": "medium",
        "yolov8_l_mobilenet_v2": "MobileNetV2"
    }
    models = list(model_names.keys())

    fps_values = []
    max_shapes = {}
    for model in models:
        max_shape = get_max_input_shape(data[device][model])
        onnx_fps_value = float(data[device][model][max_shape]['onnxruntime']['none']['dynamic'][str(batch_size)]['fps'])
        tensorrt_fps_value = float(data[device][model][max_shape]['tensorrt']['none']['dynamic'][str(batch_size)]['fps'])
        fps_values.append((model, onnx_fps_value, tensorrt_fps_value))
        max_shapes[model] = max_shape

        # Check if mAP values are the same for both backends
        # onnx_mAP_value = float(data[device][model][max_shape]['onnxruntime']['none']['dynamic'][str(batch_size)]['bbox_mAP'])
        # tensorrt_mAP_value = float(data[device][model][max_shape]['tensorrt']['none']['dynamic'][str(batch_size)]['bbox_mAP'])
        # if onnx_mAP_value != tensorrt_mAP_value:
        #     print(f"{device}, {model}, {max_shape}, {batch_size}: ort mAP {onnx_mAP_value} and trt mAP {tensorrt_mAP_value}!")
        # else:
        #     print("mAP values are the same")
        # Result: Indeed, they are


    # Sort bars by ONNX runtime FPS
    fps_values.sort(key=lambda x: x[2], reverse=True)

    labels = [f"{model_names[model]}\n" + r"$" + max_shapes[model].split("x")[0] + r" \times " + max_shapes[model].split("x")[1] + r"$" for model, _, _ in fps_values]
    onnx_fps = [fps for _, fps, _ in fps_values]
    tensorrt_fps = [fps for _, _, fps in fps_values]

    ind = range(len(models))
    width = 0.35

    ax.barh([i + width / 2 for i in ind], onnx_fps, width, label='ONNX Runtime')
    ax.barh([i - width / 2 for i in ind], tensorrt_fps, width, label='TensorRT')

    # ax.invert_yaxis()
    for i, v in enumerate(onnx_fps):
        ax.text(v + (max(onnx_fps) * 0.025), i + 0.075, f'{v:.1f}', color='black', fontsize=8)
        # ax.text(v, i + 0.05, f'{v:.1f}', color='black', fontsize=8)
    for i, v in enumerate(tensorrt_fps):
        ax.text(v + (max(onnx_fps) * 0.025), i - 0.3, f'{v:.1f}', color='black', fontsize=8)

    ax.set_xlim(0, max(tensorrt_fps) * 1.1)
    ax.set_title(f'{get_device_name(device)}\,---\,Batch Size {batch_size}', fontsize=12)
    ax.set_yticks(ind)
    ax.set_yticklabels(labels)
    ax.set_xlabel('FPS')
    ax.set_ylabel('Model')

    ax.legend()

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 5.5))
create_subplot('nano', 1, ax1)
create_subplot('agx', 32, ax2)
plt.tight_layout()

# Add suptitle
fig.suptitle("Inference Speeds: ONNX Runtime vs. TensorRT on YOLOv8 Models", y=1.02, fontsize=14)

# Save figure
figures_path = "/home/tedro/Desktop/d_projekty/bp/proj/doc/figures/"
figure_name = "onnx_vs_tensorrt_comparison.pdf"
fig.savefig(os.path.join(figures_path, figure_name), bbox_inches='tight', transparent=True)
