import json
import os
import pandas as pd


tests_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tests")
with open(os.path.join(tests_path, "test_results.json")) as f:
    data = json.load(f)

device = "agx"
model_shape = "dynamic"
batch_size = "32"

model_names = {
    "yolov8_f": "femto",
    "yolov8_p": "pico",
    "yolov8_n": "nano",
    "yolov8_s": "small",
    "yolov8_m": "medium",
    "yolov8_l_mobilenet_v2": "MobileNetV2",
}

device_names = {
    "rpi": "Raspberry PI",
    "intel_i7": "Intel Core i7-9850H",
    "mx150": "NVIDIA MX150",
    "nano": "NVIDIA Jetson Nano",
    "nx": "NVIDIA Jetson Xavier NX",
    "agx": "NVIDIA Jetson AGX Xavier",
}

model_order = {
    "medium": 0,
    "small": 1,
    "MobileNetV2": 2,
    "nano": 3,
    "pico": 4,
    "femto": 5,
}

rows = []
for model, model_data in data[device].items():
    for input_resolution, input_resolution_data in model_data.items():
        for quantization, quantization_data in input_resolution_data["tensorrt"].items():
            metrics = quantization_data[model_shape][batch_size]

            row = {
                "Model": model_names[model],
                "Input Resolution": input_resolution,
                "Quantization": quantization.replace("none", "FP32").replace("fp16", "FP16").replace("int8", "INT8"),
                "mAP": float(metrics["bbox_mAP"]),
            }
            for device_name in device_names:
                backend = "tensorrt" if device_name in ["nano", "nx", "agx"] else "onnxruntime"
                try:
                    fps = data[device_name][model][input_resolution][backend][quantization][model_shape][batch_size]["fps"]
                    row[device_names[device_name]] = float(fps)
                except KeyError as e:
                    print("WARNING: Exception ignored:")
                    from traceback import print_exc
                    print_exc()
                    if quantization not in data[device_name][model][input_resolution][backend].keys():
                        row[device_names[device_name]] = "-"
                    else:
                        row[device_names[device_name]] = "N/A"
                
            rows.append(row)

df = pd.DataFrame(rows)

# def format_fps(val):
#     if isinstance(val, str):
#         return val
#     else:
#         return "{:.2f}".format(val)
# df = df.applymap(format_fps)

df["Model"] = df["Model"].astype(pd.CategoricalDtype(categories=[k for k, v in sorted(model_order.items(), key=lambda x: x[1])], ordered=True))
df["Input Resolution"] = df["Input Resolution"].astype(pd.CategoricalDtype(categories=sorted(set(df["Input Resolution"]), key=lambda x: (-int(x.split("x")[0]), -int(x.split("x")[1]))), ordered=True))
df = df.sort_values(by=["Model", "Input Resolution"])

df["Input Resolution"] = df["Input Resolution"].apply(lambda x: "$" + x.replace('x', ' \\times ') + "$")

column_format = "lllrrrrrr"

def basic(df):

    latex_table = df.to_latex(index=False, column_format=column_format, escape=False)

    print(latex_table)

def merged(df):

    def replace_redundant_values(series):
        prev_value = None
        new_series = []
        for value in series:
            if value == prev_value:
                new_series.append('')
            else:
                new_series.append(value)
                prev_value = value
        return new_series

    df['Model'] = replace_redundant_values(df['Model'])
    df['Input Resolution'] = replace_redundant_values(df['Input Resolution'])

    latex_table = df.to_latex(index=False, escape=False, multirow=True)

    # TODO 
    # latex_table = df.to_latex(index=False, column_format="lllrrrrrr", escape=True, multirow=True)

    print(latex_table)


# basic(df)
merged(df)