import os
import json
import numpy as np
import matplotlib.pyplot as plt

tests_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tests")
with open(os.path.join(tests_path, "test_results.json")) as f:
    data = json.load(f)

plt.rcParams.update({"font.size": 10})
plt.rcParams.update({"font.family": "serif"})
plt.rcParams.update({"font.serif": "cmr10"})
plt.rcParams.update({"text.usetex": True})

model_names = {
    "yolov8_m": "medium",
    "yolov8_s": "small",
    "yolov8_l_mobilenet_v2": "MobileNetV2",
    "yolov8_n": "nano",
    "yolov8_p": "pico",
    "yolov8_f": "femto",
}
device_names = {
    "rpi": "Raspberry PI",
    "intel_i7": "Intel Core i7-9850H",
    "mx150": "MX150",
    "nano": "Jetson Nano",
    "nx": "Jetson Xavier NX",
    "agx": "Jetson AGX Xavier",
}



model_fps_data = {}

for device in ["nx", "nano", "agx"]:
    device_data = data[device]
    for model, model_specs in device_data.items():
        # Find the highest input resolution for the model
        highest_input_resolution = sorted(model_specs.keys(), key=lambda x: int(x.split("x")[0]) * int(x.split("x")[1]), reverse=True)[0]
        
        for input_resolution, input_resolution_data in model_specs.items():

            # if input_resolution != highest_input_resolution:
            #     continue

            # Only display highest input resolution for each model but also with femto 352x192 and take nano with 512x288
            if model == "yolov8_n":
                if input_resolution != "512x288":
                    continue
            elif model == "yolov8_f":
                if input_resolution != "352x192":
                    continue
            elif input_resolution != highest_input_resolution:
                continue

            backend = "tensorrt"
            quantization = "fp16"
            model_shape = "dynamic"

            if model in ["yolov8_m", "yolov8_s"] or model.startswith("yolov8_l"):
                batch_size = 8
            else:
                batch_size = 32

            fps = float(input_resolution_data[backend][quantization][model_shape][str(batch_size)]["fps"])
            model_key = f"{model_names[model]} {input_resolution}"
            if model_key not in model_fps_data:
                model_fps_data[model_key] = {}
            model_fps_data[model_key][device] = fps

sorted_model_fps_data = dict(sorted(model_fps_data.items(), key=lambda x: x[1]["agx"], reverse=True))

# fig, ax = plt.subplots(figsize=(6, len(sorted_model_fps_data) * 0.45))
# fig, ax = plt.subplots(figsize=(6, 5)) # for text (FPS) for hbars
fig, ax = plt.subplots(figsize=(6, 3.25)) # no FPS text
y_pos = np.arange(len(sorted_model_fps_data))

# light colors but C1 is close to C8!
# device_colors = {
#     "nano": "C8",
#     "nx": "C9",
#     "agx": "C1",
# }

device_colors = {
    "nano": "C0",
    "nx": "C1",
    "agx": "C2",
}

bar_width = 0.275

legend_handles = []

for i, (model_key, device_data) in enumerate(sorted_model_fps_data.items()):
    # Sort the filtered_device_data based on the device names order in the reversed device_names list
    sorted_device_data = {}
    for device_name in reversed(list(device_names.keys())):
        if device_name in device_data:
            sorted_device_data[device_name] = device_data[device_name]

    for j, (device, fps) in enumerate(sorted_device_data.items()):
        color = device_colors[device]
        bar = ax.barh(y_pos[i] + (j - 1) * bar_width, fps, bar_width,
                      label=device_names[device], color=color)
        if i == 0:  # Add legend handles only once
            legend_handles.append(bar)


        # Comment out to put FPS values to the hbars
        # if True: continue

        # Add FPS values to the right of each horizontal bar
        fps_text = f"{fps:.1f}"  # One decimal place for other devices

        x = fps * 1.04

        # Or put them at the end of the hbar
        # x = fps * (0.97 - 0.055 * len(fps_text))
        # if x < 0.1:
        #     x = fps * 1.1

        # x = 5.25

        ax.text(x, y_pos[i] + (j - 1) * bar_width - 0.015,
                fps_text, fontsize=8, va="center")

ax.set_axisbelow(True)
ax.grid(True, linestyle="--", alpha=0.3)

ax.set_yticks(y_pos)
ax.set_yticklabels([model_key.split(' ')[0] + "\n$" + model_key.split(' ')[1].replace('x', ' \\times ') + "$" for model_key in sorted_model_fps_data.keys()])

ax.set_ylabel('Model')
ax.set_xlabel('FPS (logarithmic scale)')
ax.legend(handles=list(reversed(legend_handles)), title="Devices", loc="upper right", bbox_to_anchor=(1.0, 1), fontsize=10)
ax.set_xscale('log')
xticks = [5, 10, 20, 50, 100, 200, 500, 1000, 2000]
ax.set_xlim(xticks[0])
ax.set_xticks(xticks)
ax.set_xticklabels([str(t) for t in xticks[:-1]] + [""])

plt.tight_layout()
plt.subplots_adjust(top=0.95)
fig.suptitle("Inference Speeds of Quantized YOLOv8 Models (FP16) on Jetson Devices", fontsize=12, y=1.01)

save_path = "/home/tedro/Desktop/d_projekty/bp/proj/doc/figures/"
plt.savefig(os.path.join(save_path, "devices_benchmark_fp16.pdf"), bbox_inches="tight", transparent=True)
