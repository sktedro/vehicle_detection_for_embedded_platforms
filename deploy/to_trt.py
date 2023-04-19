"""Converts an ONNX model to TensorRT format"""
import os
import argparse

import sys
repo_path = os.path.join(os.path.dirname(__file__), '..')
sys.path.append(repo_path)
import paths


# Should work but needs trtexec command
def main(onnx_filepath, trt_filepath):
    cmd = ["trtexec"]
    cmd += ["--onnx=" + onnx_filepath]
    cmd += ["--saveEngine=" + trt_filepath]
    os.system(cmd)


# Might not work
# def main(onnx_filepath, trt_filepath):
#     import onnx
#     import tensorrt as trt
#     onnx_model = onnx.load(onnx_filepath)
#     # Create TensorRT builder and network
#     builder = trt.Builder(trt.Logger())
#     network = builder.create_network()

#     # Create ONNX parser and parse the model
#     parser = trt.OnnxParser(network, builder.logger)
#     parser.parse(onnx_model.SerializeToString())

#     # Optimize the network and create an engine
#     builder.max_batch_size = 1
#     builder.max_workspace_size = 1 << 30
#     engine = builder.build_cuda_engine(network)

#     # Serialize the engine to a file
#     with open(trt_filepath, "wb") as f:
#         f.write(engine.serialize())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("work_dir",          type=str,   default=paths.working_dirpath, nargs="?",
                        help="working dirpath. Leave blank to use paths.working_dirpath")
    parser.add_argument("-i", "--input_filename",   type=str,   default=paths.deploy_onnx_filename,
                        help="input filename. Leave blank to use paths.deploy_onnx_filename")
    parser.add_argument("-o", "--output_filename",   type=str,   default=paths.deploy_trt_filename,
                        help="output filename. Leave blank to use paths.deploy_trt_filename")
    args = parser.parse_args()

    onnx_filepath = os.path.join(args.work_dir, args.input_filename)
    trt_filepath = os.path.join(args.work_dir, args.output_filename)

    assert os.path.exists(onnx_filepath), f"Deployed ONNX model does not exist at {onnx_filepath}"

    main(onnx_filepath, trt_filepath)