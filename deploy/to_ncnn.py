"""Converts an ONNX model to NCNN format"""
import os
import argparse
from subprocess import call
from mmdeploy.backend.ncnn.init_plugins import get_onnx2ncnn_path

import sys
repo_path = os.path.join(os.path.dirname(__file__), '..')
sys.path.append(repo_path)
import paths


# Something like this could work, but doesn't - get_onnx2ncnn_path() fails
# Taken from mmdeploy/backend/ncnn/onnx2ncnn.py
def main(onnx_filepath, work_dir, output_filename):
    param_filepath = os.path.join(work_dir, output_filename + '.param')
    bin_filepath = os.path.join(work_dir, output_filename + '.bin')

    onnx2ncnn_path = get_onnx2ncnn_path()
    ret_code = call([onnx2ncnn_path, onnx_filepath, param_filepath, bin_filepath])
    assert ret_code == 0, 'onnx2ncnn failed'


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("work_dir",          type=str,   default=paths.working_dirpath, nargs="?",
                        help="working dirpath. Leave blank to use paths.working_dirpath")
    parser.add_argument("-i", "--input_filename",   type=str,   default=paths.deploy_onnx_filename,
                        help="input filename. Leave blank to use paths.deploy_onnx_filename")
    parser.add_argument("-o", "--output_filename",   type=str,   default=paths.deploy_ncnn_filename,
                        help="output filename without an extension. Leave blank to use paths.deploy_ncnn_filename")
    args = parser.parse_args()

    onnx_filepath = os.path.join(args.work_dir, args.input_filename)

    assert os.path.exists(onnx_filepath), f"Deployed ONNX model does not exist at {onnx_filepath}"

    main(onnx_filepath, args.work_dir, args.output_filename)