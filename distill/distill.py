from mmrazor.models.algorithms.distill.configurable import SingleTeacherDistill
import os
from mmengine.config import Config
# from mmrazor.registry import MODELS
from mmengine.runner import Runner
from mmengine.registry import MODELS
from mmengine.registry import build_model_from_cfg
from mmengine.registry import build_from_cfg



distill_conf_dict_filepath = os.path.join(os.path.dirname(__file__), "conf.py")
distill_conf_dict = Config.fromfile(distill_conf_dict_filepath)
# distill_conf_dict = Config._dict_to_config_dict(distill_conf_dict)
# distill_conf_dict = distill_conf_dict.model
distill_conf_dict = distill_conf_dict.model.distiller

teacher_conf_filepath = '/home/xskalo01/bp/proj/configs/yolov8_m.py'
teacher_conf_dict = Config.fromfile(teacher_conf_filepath)
teacher_conf_dict = teacher_conf_dict.model
# teacher_conf_dict = build_model_from_cfg(teacher_conf_dict, MODELS)
teacher_conf_dict = build_from_cfg(teacher_conf_dict, MODELS)
# teacher_conf_dict = MODELS.build(teacher_conf_dict)
# teacher_conf_dict = Config._dict_to_config_dict(teacher_conf_filepath)

teacher_ckpt = "/home/xskalo01/bp/proj/working_dir_yolov8_dist_384_conf8/epoch_125.pth"

model_checkpoint_filename = "yolov8_n_syncbn_fast_8xb16-500e_coco_20230114_131804-88c11cdb.pth"
model_checkpoint_filepath = os.path.join("/home/xskalo01/bp/proj/", "checkpoints", model_checkpoint_filename)

architecture = dict(
    cfg_path='mmyolo::/home/xskalo01/bp/proj/configs/yolov8_n.py',
    pretrained=False,
    # model_path=model_checkpoint_filepath,
    # pretrained=True
    )

distiller = SingleTeacherDistill(distiller=distill_conf_dict,
                                 teacher=teacher_conf_dict,
                                 teacher_ckpt=teacher_ckpt,
                                 architecture=architecture
)

distiller.train()



# distiller = Runner.from_cfg(distill_conf_dict)