"""
Merges all test results into a single file. Tests are expected in proj/tests/
folder and each test should be a folder containing test_all_results.json file

Example input and output:
```
"jetson_nano": {
    "yolov8_f": {
        "352x192": {
            "onnxruntime": {
                "int8": {
                    "dynamic": {
                        "1": {
                            "bbox_mAP": "0.2590",
                            "bbox_mAP_50": "0.4300",
                            "bbox_mAP_75": "0.2750",
                            "bbox_mAP_s": "0.0540",
                            "bbox_mAP_m": "0.2270",
                            "bbox_mAP_l": "0.3700"
                            ...
    ```
"""
import os
import json as json_lib
from pprint import pprint


def parse_json(f, results):
    with open(f) as json_file:
        json = json_lib.load(json_file)

    for device_name in json:
        if device_name not in results:
            results[device_name] = {}

        for model_name in json[device_name]:
            if model_name not in results[device_name]:
                results[device_name][model_name] = {}

            for input_shape in json[device_name][model_name]:
                if input_shape not in results[device_name][model_name]:
                    results[device_name][model_name][input_shape] = {}

                for backend in json[device_name][model_name][input_shape]:
                    if backend not in results[device_name][model_name][input_shape]:
                        results[device_name][model_name][input_shape][backend] = {}

                    for quant in json[device_name][model_name][input_shape][backend]:
                        if quant not in results[device_name][model_name][input_shape][backend]:
                            results[device_name][model_name][input_shape][backend][quant] = {}

                        for model_shape in json[device_name][model_name][input_shape][backend][quant]:
                            if model_shape not in results[device_name][model_name][input_shape][backend][quant]:
                                results[device_name][model_name][input_shape][backend][quant][model_shape] = {}

                            for batch_size in json[device_name][model_name][input_shape][backend][quant][model_shape]:
                                if batch_size not in results[device_name][model_name][input_shape][backend][quant][model_shape]:
                                    results[device_name][model_name][input_shape][backend][quant][model_shape][batch_size] \
                                            = json[device_name][model_name][input_shape][backend][quant][model_shape][batch_size]
                                else:
                                    print(f"Duplicate entry found: "
                                        "{device_name}, {model_name}, {input_shape}, {backend}, {quant}, {model_shape}, {batch_size}: "
                                        "{json[device_name][model_name][input_shape][backend][quant][model_shape][batch_size]}")
                                    print(f"In {f}")


def main():

    tests_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tests")
    if not os.path.exists(tests_path):
        print(f"{tests_path}, expected to contain folders with 'test_all_results.json' files does not exist")
        exit(1)

    # Get all test result files
    files = []
    for d in os.listdir(tests_path):
        d = os.path.join(tests_path, d)
        if os.path.isdir(d):
            if os.path.exists(os.path.join(tests_path, d, "test_all_results.json")):
                files.append(os.path.join(tests_path, d, "test_all_results.json"))
    files.sort()

    print(f"{len(files)} test files found to read:")
    pprint(files)

    results = {}
    for f in files:
        parse_json(f, results)

    results_json = json_lib.dumps(results, indent=2)

    # print("Results:")
    # print(results_json)

    with open(os.path.join(tests_path, "test_results.json"), "w") as f:
        f.write(results_json)

    print("All done")


if __name__ == "__main__":
    main()
