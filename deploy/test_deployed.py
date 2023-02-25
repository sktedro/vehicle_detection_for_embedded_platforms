import subprocess

import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import paths

assert paths.last_checkpoint_filepath != None
assert os.path.exists(paths.deploy_config_filepath)
assert os.path.exists(paths.model_config_filepath)
assert os.path.exists(paths.last_checkpoint_filepath)


cmd = ["/home/xskalo01/.localpython/bin/python3", "/home/xskalo01/bp/mmdeploy/tools/test.py"]
cmd += [paths.deploy_config_filepath] # Deploy config
cmd += [paths.model_config_filepath] # Model config
cmd += ["--model", os.path.join(paths.working_dirpath, paths.deploy_onnx_filename)]
cmd += ["--work-dir", paths.working_dirpath]
cmd += ["--show-dir", os.path.join(paths.working_dirpath, "deployed_test/")]
cmd += ["--log2file", os.path.join(paths.working_dirpath, "deployed_test_log/")]
cmd += ["--device", "cpu"]
# cmd += ["--speed-test"]

print(cmd)
ret = subprocess.call(cmd)
print("Returned:", ret)