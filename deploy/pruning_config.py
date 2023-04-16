_base_ = [
    '/home/xskalo01/bp/proj/configs/yolov8_m.py',
]


default_scope = 'mmrazor'

_base_.model._scope_ = 'mmyolo'

_base_.model = dict(
    _scope_='mmrazor',
    type='DCFF',
    architecture=_base_.model,
    mutator_cfg=dict(
        type='DCFFChannelMutator',
        channel_unit_cfg=dict(
            type='DCFFChannelUnit', default_args=dict(choice_mode='ratio')),
        parse_cfg=dict(
            type='ChannelAnalyzer',
            demo_input=(1, 3, 224, 224),
            tracer_type='FxTracer')),
    # target_pruning_ratio=target_pruning_ratio,
    step_freq=1,
    linear_schedule=False)

model_wrapper = dict(
    type='mmcv.MMDistributedDataParallel', find_unused_parameters=True)

val_cfg = dict(_delete_=True, type='mmrazor.ItePruneValLoop')
