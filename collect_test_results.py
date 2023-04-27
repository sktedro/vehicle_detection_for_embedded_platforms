"""
Collects results of all tests of which logs are available
"""
import os
import json
import re
from pprint import pprint


def main():
    # Get all work dirs
    proj_path = os.path.dirname(os.path.abspath(__file__))
    work_dirpaths = []
    for f in os.listdir(proj_path):
        if os.path.isdir(f):
            if f.startswith("working_dir"):
                work_dirpaths.append(os.path.join(proj_path, f))
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

    model_name_regex = r"^working_dir_(.*)_(\d+)x(\d+)"
    fps_regex = r".*times per count: [\d.]* ms, ([\d.]*) FPS"

    results = {}
    for f in test_files:
        work_dirname = os.path.basename(os.path.dirname(f))
        model_name, w, h = re.match(model_name_regex, work_dirname).groups()
        input_size = f"{w}x{h}"

        # details example: [test onnxruntime dynamic batch onnx batch2]
        details = os.path.basename(f).split(".")[0].split("_")
        backend = details[1]
        shape = details[2]
        batch = details[-1].replace("batch", "")

        with open(f) as fd:
            content = fd.readlines()

            test_results = {}

            # Get mAP values from the last line in the file
            last_line = content[-1].replace("  ", " ").replace("  ", " ").split(" ")
            for i in range(len(last_line)):
                if "coco/" in last_line[i]:
                    mAP_name = last_line[i].split("/")[1].split(":")[0]
                    mAP_val = last_line[i+1]
                    test_results[mAP_name] = mAP_val
            for s in ["bbox_mAP", "bbox_mAP_50", "bbox_mAP_75", "bbox_mAP_s", "bbox_mAP_m", "bbox_mAP_l"]:
                assert s in list(test_results.keys()), f"{s} missing in the last line of the file {f}"

            # Get average FPS from the last line containing "FPS" in the file
            for line in reversed(content):
                if line.endswith("FPS\n"):
                    test_results["fps"] = re.match(fps_regex, line).groups()[0]
                    break
            else:
                raise Exception(f"FPS not found in file {f}")

        # Save the data
        if model_name not in results:
            results[model_name] = {}
        if input_size not in results[model_name]:
            results[model_name][input_size] = {}
        if backend not in results[model_name][input_size]:
            results[model_name][input_size][backend] = {}
        if shape not in results[model_name][input_size][backend]:
            results[model_name][input_size][backend][shape] = {}
        results[model_name][input_size][backend][shape][batch] = test_results

    results_json = json.dumps(results, indent=2)

    print("Results:")
    print(results_json)

    with open("test_all_results.json", "w") as f:
        f.write(results_json)


if __name__ == "__main__":
    main()