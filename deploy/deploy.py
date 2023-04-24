# Copyright (c) OpenMMLab. All rights reserved.
# File taken from mmdeploy/tools/ and slightly modified by Patrik Skaloš
# TODO refactor the code because this is humiliating
import argparse
import logging
import os
import os.path as osp
import sys
from functools import partial

import mmengine
import torch.multiprocessing as mp
from torch.multiprocessing import Process, set_start_method

from mmdeploy.apis import (create_calib_input_data, extract_model,
                           get_predefined_partition_cfg, torch2onnx,
                           torch2torchscript, visualize_model)
from mmdeploy.apis.core import PIPELINE_MANAGER
from mmdeploy.apis.utils import to_backend
from mmdeploy.utils import (IR, Backend, get_backend, get_calib_filename,
                            get_ir_config, get_partition_config,
                            get_root_logger, load_config, target_wrapper)

import common as deploy_common

repo_path = os.path.join(os.path.dirname(__file__), '..')
sys.path.append(repo_path)
import paths
# from dataset import common


def create_process(name, target, args, kwargs, ret_value=None):
    logger = get_root_logger()
    logger.info(f'{name} start.')
    log_level = logger.level

    wrap_func = partial(target_wrapper, target, log_level, ret_value)

    process = Process(target=wrap_func, args=args, kwargs=kwargs)
    process.start()
    process.join()

    if ret_value is not None:
        if ret_value.value != 0:
            logger.error(f'{name} failed.')
            exit(1)
        else:
            logger.info(f'{name} success.')


def torch2ir(ir_type: IR):
    """Return the conversion function from torch to the intermediate
    representation.

    Args:
        ir_type (IR): The type of the intermediate representation.
    """
    if ir_type == IR.ONNX:
        return torch2onnx
    elif ir_type == IR.TORCHSCRIPT:
        return torch2torchscript
    else:
        raise KeyError(f'Unexpected IR type {ir_type}')


def main(args):

    assert os.path.exists(args.work_dir)

    # Find deploy config
    if not os.path.exists(args.deploy_cfg):
        if not args.deploy_cfg.endswith(".py"):
            args.deploy_cfg += ".py"
        if os.path.exists(os.path.join(paths.proj_path, args.deploy_cfg)):
            args.deploy_cfg = os.path.join(paths.proj_path, args.deploy_cfg)
        if os.path.exists(os.path.join(paths.proj_path, "deploy", args.deploy_cfg)):
            args.deploy_cfg = os.path.join(paths.proj_path, "deploy", args.deploy_cfg)
    assert os.path.exists(args.deploy_cfg), f"Deploy config not found anywhere ({args.deploy_cfg})"

    # Get the filepath of the model configuration file
    model_config_filepath = paths.get_config_from_working_dirpath(args.work_dir)

    if isinstance(args.epoch, int):
        checkpoint_filepath = os.path.join(args.work_dir, f"epoch_{args.epoch}.pth")
    else:
        checkpoint_filepath = paths.get_best_checkpoint_filepath(args.work_dir)
        args.epoch = int(checkpoint_filepath.split("_")[-1].split(".")[0])

    # Print args
    for name, value in vars(args).items():
        print(name + ":", value)
    print("model config:", model_config_filepath)
    print("checkpoint:", checkpoint_filepath)

    img = os.path.join(paths.proj_path, "deploy", "demo.jpg")

    set_start_method('spawn', force=True)
    logger = get_root_logger()
    log_level = logging.getLevelName(args.log_level)
    logger.setLevel(log_level)

    pipeline_funcs = [
        torch2onnx, torch2torchscript, extract_model, create_calib_input_data
    ]
    PIPELINE_MANAGER.enable_multiprocess(True, pipeline_funcs)
    PIPELINE_MANAGER.set_log_level(log_level, pipeline_funcs)

    # Get model and deploy configs
    model_cfg = load_config(model_config_filepath)
    # TODO TODO TODO use this in inference scripts
    deploy_cfg = deploy_common.get_deploy_config(args.deploy_cfg, model_cfg, args.output_filename)

    # create work_dir if not
    mmengine.mkdir_or_exist(osp.abspath(args.work_dir))

    ret_value = mp.Value('d', 0, lock=False)

    # convert to IR
    ir_config = get_ir_config(deploy_cfg)
    ir_save_file = ir_config['save_file']
    ir_type = IR.get(ir_config['type'])
    torch2ir(ir_type)(
        img,
        args.work_dir,
        ir_save_file,
        deploy_cfg,
        model_config_filepath,
        checkpoint_filepath,
        device=args.device)

    # convert backend
    ir_files = [osp.join(args.work_dir, ir_save_file)]

    # partition model
    partition_cfgs = get_partition_config(deploy_cfg)

    if partition_cfgs is not None:

        if 'partition_cfg' in partition_cfgs:
            partition_cfgs = partition_cfgs.get('partition_cfg', None)
        else:
            assert 'type' in partition_cfgs
            partition_cfgs = get_predefined_partition_cfg(
                deploy_cfg, partition_cfgs['type'])

        origin_ir_file = ir_files[0]
        ir_files = []
        for partition_cfg in partition_cfgs:
            save_file = partition_cfg['save_file']
            save_path = osp.join(args.work_dir, save_file)
            start = partition_cfg['start']
            end = partition_cfg['end']
            dynamic_axes = partition_cfg.get('dynamic_axes', None)

            extract_model(
                origin_ir_file,
                start,
                end,
                dynamic_axes=dynamic_axes,
                save_file=save_path)

            ir_files.append(save_path)

    # calib data
    calib_filename = get_calib_filename(deploy_cfg)
    if calib_filename is not None:
        calib_path = osp.join(args.work_dir, calib_filename)
        create_calib_input_data(
            calib_path,
            deploy_cfg,
            model_config_filepath,
            checkpoint_filepath,
            dataset_cfg=args.calib_dataset_cfg,
            dataset_type='val',
            device=args.device)

    backend_files = ir_files
    # convert backend
    backend = get_backend(deploy_cfg)

    # convert to backend
    PIPELINE_MANAGER.set_log_level(log_level, [to_backend])
    if backend == Backend.TENSORRT:
        PIPELINE_MANAGER.enable_multiprocess(True, [to_backend])
    backend_files = to_backend(
        backend,
        ir_files,
        work_dir=args.work_dir,
        deploy_cfg=deploy_cfg,
        log_level=log_level,
        device=args.device)

    # ncnn quantization
    if backend == Backend.NCNN and args.quant:
        from onnx2ncnn_quant_table import get_table

        from mmdeploy.apis.ncnn import get_quant_model_file, ncnn2int8
        model_param_paths = backend_files[::2]
        model_bin_paths = backend_files[1::2]
        backend_files = []
        for onnx_path, model_param_path, model_bin_path in zip(
                ir_files, model_param_paths, model_bin_paths):

            deploy_cfg, model_cfg = load_config(deploy_cfg,
                                                model_config_filepath)
            quant_onnx, quant_table, quant_param, quant_bin = get_quant_model_file(  # noqa: E501
                onnx_path, args.work_dir)

            create_process(
                'ncnn quant table',
                target=get_table,
                args=(onnx_path, deploy_cfg, model_cfg, quant_onnx,
                      quant_table, args.quant_image_dir, args.device),
                kwargs=dict(),
                ret_value=ret_value)

            create_process(
                'ncnn_int8',
                target=ncnn2int8,
                args=(model_param_path, model_bin_path, quant_table,
                      quant_param, quant_bin),
                kwargs=dict(),
                ret_value=ret_value)
            backend_files += [quant_param, quant_bin]

    extra = dict(
        backend=backend,
        output_file=osp.join(args.work_dir, f'output_{backend.value}.jpg'),
        show_result=False)

    # get backend inference result, try render
    if args.visualize:
        create_process(
            f'visualize {backend.value} model',
            target=visualize_model,
            args=(model_config_filepath, deploy_cfg, backend_files, img,
                args.device),
            kwargs=extra,
            ret_value=ret_value)

        # get pytorch model inference result, try visualize if possible
        create_process(
            'visualize pytorch model',
            target=visualize_model,
            args=(model_config_filepath, deploy_cfg, [checkpoint_filepath],
                img, args.device),
            kwargs=dict(
                backend=Backend.PYTORCH,
                output_file=osp.join(args.work_dir, 'output_pytorch.jpg'),
                show_result=False),
            ret_value=ret_value)
    logger.info('All process success.')


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("deploy_cfg",                type=str,
                        help="Deploy config filepath. Can be relative to deploy/ folder or relative to the project folder")
    parser.add_argument("work_dir",                  type=str,   default=paths.working_dirpath, nargs="?",
                        help="working dirpath. Leave blank to use paths.working_dirpath")
    parser.add_argument("-e", "--epoch",             type=int,
                        help="epoch number to use. Leave blank to use best based on COCO metric")
    parser.add_argument("-o", "--output-filename",   type=str,   default="end2end",
                        help="filename for the deployed model (without extension). Default end2end")
    parser.add_argument("-d", '--device',            type=str,   default='cpu',
                        help='device used for conversion. Default cpu')
    parser.add_argument("-v", '--visualize',
                        help='visualize the models (to compare the pytorch model and the backend model)', action='store_true')
    parser.add_argument("-q", '--quant',
                        help='quantize model to low bit', action='store_true')
    parser.add_argument('--calib-dataset-cfg',                   default=None,
                        help='dataset config path used to calibrate in int8 mode. If not specified, it will use "val" dataset in model config instead')
    parser.add_argument('--quant-image-dir',                     default=None,
                        help='image directory for model quantization')
    parser.add_argument('--log-level',                           default='INFO',
                        help='set log level',
                        choices=list(logging._nameToLevel.keys()))
    args = parser.parse_args()

    return args


if __name__ == '__main__':
    main(parse_args())
