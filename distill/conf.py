# import os
# import sys

# No way to find out the path of this file since mmdetection does shady things
# with it, so we need an absolute filepath of the repository
# repo_path = "/home/xskalo01/bp/proj/"
# sys.path.append(repo_path)
# from paths import paths
# import dataset.common as common

_base_ = [
    # paths.model_config_filepath, is equal to:
    "/home/xskalo01/bp/proj/configs/yolov8_n.py",
]

# TODO:

# https://github.com/open-mmlab/mmrazor/issues/448

teacher_ckpt = "/home/xskalo01/bp/proj/working_dir_yolov8_dist_384_conf8/epoch_125.pth", # TODO

model = dict(
    _scope_='mmrazor',
    type='FpnTeacherDistill',
    architecture=dict(
        cfg_path='/home/xskalo01/bp/proj/configs/yolov8_n.py', # TODO
        pretrained=True),
    teacher=dict(
        cfg_path='/home/xskalo01/bp/proj/configs/yolov8_m.py', # TODO
        pretrained=True),
    teacher_ckpt=teacher_ckpt,
    distiller=dict(
        type='ConfigurableDistiller',
        student_recorders=dict(fpn=dict(type='ModuleOutputs', source='neck')),
        teacher_recorders=dict(fpn=dict(type='ModuleOutputs', source='neck')),
        distill_losses=dict(
            loss_pkd_fpn0=dict(type='PKDLoss', loss_weight=6),
            loss_pkd_fpn1=dict(type='PKDLoss', loss_weight=6),
            loss_pkd_fpn2=dict(type='PKDLoss', loss_weight=6),
            loss_pkd_fpn3=dict(type='PKDLoss', loss_weight=6)),
        loss_forward_mappings=dict(
            loss_pkd_fpn0=dict(
                preds_S=dict(from_student=True, recorder='fpn', data_idx=0),
                preds_T=dict(from_student=False, recorder='fpn', data_idx=0)),
            loss_pkd_fpn1=dict(
                preds_S=dict(from_student=True, recorder='fpn', data_idx=1),
                preds_T=dict(from_student=False, recorder='fpn', data_idx=1)),
            loss_pkd_fpn2=dict(
                preds_S=dict(from_student=True, recorder='fpn', data_idx=2),
                preds_T=dict(from_student=False, recorder='fpn', data_idx=2)),
            loss_pkd_fpn3=dict(
                preds_S=dict(from_student=True, recorder='fpn', data_idx=3),
                preds_T=dict(from_student=False, recorder='fpn',
                             data_idx=3)))))

find_unused_parameters = True

val_cfg = dict(_delete_=True, type='mmrazor.SingleTeacherDistillValLoop')
