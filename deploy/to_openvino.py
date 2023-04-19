"""Converts an ONNX model to OpenVINO format"""
import os
import argparse

import sys
repo_path = os.path.join(os.path.dirname(__file__), '..')
sys.path.append(repo_path)
import paths


def main(onnx_filepath, openvino_filepath):
    ...


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("work_dir",          type=str,   default=paths.working_dirpath, nargs="?",
                        help="working dirpath. Leave blank to use paths.working_dirpath")
    parser.add_argument("-i", "--input_filename",   type=str,   default=paths.deploy_onnx_filename,
                        help="input filename. Leave blank to use paths.deploy_onnx_filename")
    parser.add_argument("-o", "--output_filename",   type=str,   default=paths.deploy_openvino_filename,
                        help="output filename. Leave blank to use paths.deploy_openvino_filename")
    args = parser.parse_args()

    onnx_filepath = os.path.join(args.work_dir, args.input_filename)
    openvino_filepath = os.path.join(args.work_dir, args.output_filename)

    assert os.path.exists(onnx_filepath), f"Deployed ONNX model does not exist at {onnx_filepath}"

    main(onnx_filepath, openvino_filepath)