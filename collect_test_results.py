"""
Collects results of all tests of which logs are available

Example output:
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
import json
import re
import sys
from pprint import pprint


model_name_regex = r"^working_dir_(.*)_(\d+)x(\d+)"
fps_regex = r".*times per count: [\d.]* ms, ([\d.]*) FPS"


def parse_file(f, results):
    work_dirname = os.path.basename(os.path.dirname(f))
    model_name, w, h = re.match(model_name_regex, work_dirname).groups()
    input_size = f"{w}x{h}"

    # details example: [test onnxruntime dynamic batch onnx batch2]
    details = os.path.basename(f).split(".")[0].split("_")
    backend = details[1]
    shape = details[2]
    batch = details[-1].replace("batch", "")
    if "int8" in os.path.basename(f):
        quant = "int8"
    elif "fp16" in os.path.basename(f):
        quant = "fp16"
    else:
        quant = "none" # TODO replace by fp32

    with open(f) as fd:
        content = fd.readlines()
        if content == []:
            raise Exception(f"File {f} is empty")

        test_results = {}

        # Get mAP values from the last line in the file
        last_line = content[-1].replace("  ", " ").replace("  ", " ").split(" ")
        for i in range(len(last_line)):
            if "coco/" in last_line[i]:
                mAP_name = last_line[i].split("/")[1].split(":")[0]
                mAP_val = last_line[i+1]
                test_results[mAP_name] = mAP_val
        for s in ["bbox_mAP", "bbox_mAP_50", "bbox_mAP_75", "bbox_mAP_s", "bbox_mAP_m", "bbox_mAP_l"]:
            if s not in list(test_results.keys()): 
                print(f"WARNING: {s} missing in the last line of the file {f}")

        # Get average FPS from the last line containing "FPS" in the file
        for line in reversed(content):
            if line.endswith("FPS\n"):
                test_results["fps"] = re.match(fps_regex, line).groups()[0]
                break
        else:
            print(f"WARNING: FPS not found in file {f}")

    # Save the data
    if model_name not in results:
        results[model_name] = {}
    if input_size not in results[model_name]:
        results[model_name][input_size] = {}
    if backend    not in results[model_name][input_size]:
        results[model_name][input_size][backend] = {}
    if quant      not in results[model_name][input_size][backend]:
        results[model_name][input_size][backend][quant] = {}
    if shape      not in results[model_name][input_size][backend][quant]:
        results[model_name][input_size][backend][quant][shape] = {}

    if batch in results[model_name][input_size][backend][quant][shape]:
        print("WARNING: Duplicate entry found for:", model_name, input_size, backend, quant, shape, batch)

    results[model_name][input_size][backend][quant][shape][batch] = test_results


def main():
    assert len(sys.argv) > 1, "Please provide the test device as an argument to this script"
    device = sys.argv[1]

    if len(sys.argv) == 3:
        root_dir = sys.argv[2]
        print(f"Using {root_dir} as root dir")
    else:
        root_dir = os.path.dirname(os.path.abspath(__file__))

    # Get all work dirs
    work_dirpaths = []
    for f in os.listdir(root_dir):
        if os.path.isdir(f):
            if f.startswith("working_dir"):
                work_dirpaths.append(os.path.join(root_dir, f))
    work_dirpaths.sort()

    print(f"{len(work_dirpaths)} working dirs found to read")

    # Get all test logs from work dirs
    test_files = []
    for d in work_dirpaths:
        for f in os.listdir(d):
            if f.startswith("test") and f.endswith(".log"):
                test_files.append(os.path.join(d, f))
    test_files.sort()

    print(f"{len(test_files)} test files found to read")

    print("Test files:")
    pprint(test_files)

    exceptions = []
    results = {}
    for f in test_files:
        try:
            parse_file(f, results)
        except Exception as e:
            print(f"Exception occurred when parsing file {f}: {str(e)}. Ignoring")
            exceptions.append(str(e))

    # Put the device name at the top
    results = {
        device: results,
    }

    results_json = json.dumps(results, indent=2)

    # print("Results:")
    # print(results_json)

    with open(os.path.join(root_dir, "test_all_results.json"), "w") as f:
        f.write(results_json)

    if exceptions == []:
        print(f"All done, no exceptions")
    else:
        print(f"All done, {len(exceptions)} exceptions:")
        for e in exceptions:
            print(e)


if __name__ == "__main__":
    main()
