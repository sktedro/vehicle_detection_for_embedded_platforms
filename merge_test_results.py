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
                            "bbox_mAP_l": "0.3700",
                            "fps": "30.5"
                            ...
    ```


Better explaination of the format:
```
data = {
    "device": {
        "model": {
            "input_shape": {
                "backend": {
                    "quantization": {
                        "model_shape": {
                            "batch_size": {
                                "bbox_mAP": float number in a string,
                                "bbox_mAP_50": float number in a string,
                                "bbox_mAP_75": float number in a string,
                                "bbox_mAP_s": float number in a string,
                                "bbox_mAP_m": float number in a string,
                                "bbox_mAP_l": float number in a string,
                                "fps": float number in a string
                                ...
```
"""
import os
import json as json
from pprint import pprint


def parse(input, output):
    for key in input:
        try:
            if isinstance(input[key], dict):
                if key not in output:
                    output[key] = {}
                parse(input[key], output[key])
            elif key in output:
                print(f"Duplicate entry found!")
                raise Exception
            else:
                output[key] = input[key]
        except Exception as e:
            print(f"{key}", end=" ")
            raise


def parse_json(filepath, results):

    with open(filepath) as f:
        data = json.load(f)

    try:
        parse(data, results)
    except:
        print(filepath)
        raise


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

    results_json = json.dumps(results, indent=2)

    # print("Results:")
    # print(results_json)

    with open(os.path.join(tests_path, "test_results.json"), "w") as f:
        f.write(results_json)

    print("All done")


if __name__ == "__main__":
    main()
