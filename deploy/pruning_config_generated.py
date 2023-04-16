repo_path = '/home/xskalo01/bp/proj/'
num_gpus = 4
img_scale = (288, 512)
max_epochs = 1
warmup_epochs = 5
val_interval = 5
save_epoch_intervals = 5
custom_imports = dict(imports=['custom_modules'], allow_failed_imports=False)
default_scope = 'mmyolo'
deepen_factor = 0.33
widen_factor = 0.25
load_from = '/home/xskalo01/bp/proj/working_dir_yolov8_n_conf8_512x288/epoch_300.pth'
resume = True
max_keep_ckpts = 100
pre_trained_model_batch_size_per_gpu = 16
train_batch_size_per_gpu = 192
train_num_workers = 16
val_batch_size_per_gpu = 1
val_num_workers = 16
test_batch_size_per_gpu = 1
test_num_workers = 16
base_lr = 0.0001
lr_factor = 0.01
metainfo = dict(
    classes=('bicycle', 'motorcycle', 'car', 'transporter', 'bus', 'truck',
             'trailer', 'unknown', 'mask'))
num_classes = 9
work_dir = '/home/xskalo01/bp/proj/working_dir_yolov8_n_conf8_512x288_prune'
file_client_args = dict(backend='disk')
min_gt_bbox_wh = (8, 8)
pad_val = 114
train_pipeline = [
    dict(type='LoadImageFromFile', file_client_args=dict(backend='disk')),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='mmdet.Resize', scale=(288, 512), keep_ratio=True),
    dict(
        type='mmdet.Pad',
        pad_to_square=False,
        size=(512, 288),
        pad_val=dict(img=(114, 114, 114))),
    dict(
        type='YOLOv5RandomAffine',
        scaling_ratio_range=None,
        max_translate_ratio=0.05,
        max_rotate_degree=5,
        max_shear_degree=3,
        border_val=(114, 114, 114)),
    dict(
        type='mmdet.CustomCutOut',
        prob=0.05,
        cutout_area=(0.05, 0.35),
        random_pixels=True),
    dict(
        type='mmdet.CutOut',
        n_holes=None,
        cutout_shape=None,
        fill_in=(114, 114, 114)),
    dict(
        type='mmdet.Albu',
        transforms=[
            dict(type='Blur', p=0.01),
            dict(type='MedianBlur', p=0.01),
            dict(type='ToGray', p=0.01),
            dict(type='CLAHE', p=0.01)
        ],
        bbox_params=dict(
            type='BboxParams',
            format='pascal_voc',
            label_fields=['gt_bboxes_labels', 'gt_ignore_flags']),
        keymap=dict(img='image', gt_bboxes='bboxes')),
    dict(type='YOLOv5HSVRandomAug'),
    dict(type='mmdet.RandomFlip', prob=0.5),
    dict(type='mmdet.PhotoMetricDistortion'),
    dict(
        type='mmdet.FilterAnnotations',
        min_gt_bbox_wh=(8, 8),
        keep_empty=False),
    dict(
        type='mmdet.PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'flip',
                   'flip_direction'))
]
train_pipeline_stage2 = [
    dict(type='LoadImageFromFile', file_client_args=dict(backend='disk')),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='mmdet.Resize', scale=(288, 512), keep_ratio=True),
    dict(
        type='mmdet.Pad',
        pad_to_square=False,
        size=(512, 288),
        pad_val=dict(img=(114, 114, 114))),
    dict(
        type='YOLOv5RandomAffine',
        max_rotate_degree=0,
        max_shear_degree=0,
        scaling_ratio_range=(0.9, 1.2),
        max_aspect_ratio=100,
        border_val=(114, 114, 114)),
    dict(
        type='mmdet.Albu',
        transforms=[
            dict(type='Blur', p=0.01),
            dict(type='MedianBlur', p=0.01),
            dict(type='ToGray', p=0.01),
            dict(type='CLAHE', p=0.01)
        ],
        bbox_params=dict(
            type='BboxParams',
            format='pascal_voc',
            label_fields=['gt_bboxes_labels', 'gt_ignore_flags']),
        keymap=dict(img='image', gt_bboxes='bboxes')),
    dict(type='YOLOv5HSVRandomAug'),
    dict(type='mmdet.RandomFlip', prob=0.5),
    dict(type='mmdet.PhotoMetricDistortion'),
    dict(
        type='mmdet.FilterAnnotations',
        min_gt_bbox_wh=(8, 8),
        keep_empty=False),
    dict(
        type='mmdet.PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'flip',
                   'flip_direction'))
]
train_datasets_repeats = dict({
    'mio-tcd': 1,
    'aau': 3,
    'ndis': 25,
    'mtid': 5,
    'visdrone_det': 4,
    'detrac': 1
})
train_datasets_scaling_ratios = dict({
    'mio-tcd': (1.0, 1.1),
    'aau': (0.9, 1.1),
    'ndis': (0.9, 1.5),
    'mtid': (0.9, 2),
    'visdrone_det': (1.5, 2.5),
    'detrac': (0.8, 1)
})
train_dataset_cutout_vals = dict({
    'mio-tcd': [4, (26, 26)],
    'aau': [8, (10, 10)],
    'ndis': [12, (20, 20)],
    'mtid': [12, (10, 10)],
    'visdrone_det': [20, (8, 8)],
    'detrac': [6, (22, 22)]
})
train_dataloader = dict(
    batch_size=192,
    num_workers=16,
    persistent_workers=True,
    pin_memory=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    collate_fn=dict(type='yolov5_collate'),
    dataset=dict(
        type='ConcatDataset',
        datasets=[
            dict(
                type='RepeatDataset',
                times=1,
                dataset=dict(
                    metainfo=dict(
                        classes=('bicycle', 'motorcycle', 'car', 'transporter',
                                 'bus', 'truck', 'trailer', 'unknown',
                                 'mask')),
                    type='YOLOv5CocoDataset',
                    ann_file=
                    '/home/xskalo01/datasets/MIO-TCD/MIO-TCD-Localization/gt_processed_train.json',
                    data_prefix=dict(
                        img=
                        '/home/xskalo01/datasets/MIO-TCD/MIO-TCD-Localization/'
                    ),
                    data_root=
                    '/home/xskalo01/datasets/MIO-TCD/MIO-TCD-Localization/',
                    filter_cfg=dict(filter_empty_gt=False, min_size=32),
                    pipeline=[
                        dict(
                            type='LoadImageFromFile',
                            file_client_args=dict(backend='disk')),
                        dict(type='LoadAnnotations', with_bbox=True),
                        dict(
                            type='mmdet.Resize',
                            scale=(288, 512),
                            keep_ratio=True),
                        dict(
                            type='mmdet.Pad',
                            pad_to_square=False,
                            size=(512, 288),
                            pad_val=dict(img=(114, 114, 114))),
                        dict(
                            type='YOLOv5RandomAffine',
                            scaling_ratio_range=(1.0, 1.1),
                            max_translate_ratio=0.05,
                            max_rotate_degree=5,
                            max_shear_degree=3,
                            border_val=(114, 114, 114)),
                        dict(
                            type='mmdet.CustomCutOut',
                            prob=0.05,
                            cutout_area=(0.05, 0.35),
                            random_pixels=True),
                        dict(
                            type='mmdet.CutOut',
                            n_holes=4,
                            cutout_shape=(26, 26),
                            fill_in=(114, 114, 114)),
                        dict(
                            type='mmdet.Albu',
                            transforms=[
                                dict(type='Blur', p=0.01),
                                dict(type='MedianBlur', p=0.01),
                                dict(type='ToGray', p=0.01),
                                dict(type='CLAHE', p=0.01)
                            ],
                            bbox_params=dict(
                                type='BboxParams',
                                format='pascal_voc',
                                label_fields=[
                                    'gt_bboxes_labels', 'gt_ignore_flags'
                                ]),
                            keymap=dict(img='image', gt_bboxes='bboxes')),
                        dict(type='YOLOv5HSVRandomAug'),
                        dict(type='mmdet.RandomFlip', prob=0.5),
                        dict(type='mmdet.PhotoMetricDistortion'),
                        dict(
                            type='mmdet.FilterAnnotations',
                            min_gt_bbox_wh=(8, 8),
                            keep_empty=False),
                        dict(
                            type='mmdet.PackDetInputs',
                            meta_keys=('img_id', 'img_path', 'ori_shape',
                                       'img_shape', 'flip', 'flip_direction'))
                    ])),
            dict(
                type='RepeatDataset',
                times=3,
                dataset=dict(
                    metainfo=dict(
                        classes=('bicycle', 'motorcycle', 'car', 'transporter',
                                 'bus', 'truck', 'trailer', 'unknown',
                                 'mask')),
                    type='YOLOv5CocoDataset',
                    ann_file=
                    '/home/xskalo01/datasets/AAU/gt_processed_train.json',
                    data_prefix=dict(img='/home/xskalo01/datasets/AAU/'),
                    data_root='/home/xskalo01/datasets/AAU/',
                    filter_cfg=dict(filter_empty_gt=False, min_size=32),
                    pipeline=[
                        dict(
                            type='LoadImageFromFile',
                            file_client_args=dict(backend='disk')),
                        dict(type='LoadAnnotations', with_bbox=True),
                        dict(
                            type='mmdet.Resize',
                            scale=(288, 512),
                            keep_ratio=True),
                        dict(
                            type='mmdet.Pad',
                            pad_to_square=False,
                            size=(512, 288),
                            pad_val=dict(img=(114, 114, 114))),
                        dict(
                            type='YOLOv5RandomAffine',
                            scaling_ratio_range=(0.9, 1.1),
                            max_translate_ratio=0.05,
                            max_rotate_degree=5,
                            max_shear_degree=3,
                            border_val=(114, 114, 114)),
                        dict(
                            type='mmdet.CustomCutOut',
                            prob=0.05,
                            cutout_area=(0.05, 0.35),
                            random_pixels=True),
                        dict(
                            type='mmdet.CutOut',
                            n_holes=8,
                            cutout_shape=(10, 10),
                            fill_in=(114, 114, 114)),
                        dict(
                            type='mmdet.Albu',
                            transforms=[
                                dict(type='Blur', p=0.01),
                                dict(type='MedianBlur', p=0.01),
                                dict(type='ToGray', p=0.01),
                                dict(type='CLAHE', p=0.01)
                            ],
                            bbox_params=dict(
                                type='BboxParams',
                                format='pascal_voc',
                                label_fields=[
                                    'gt_bboxes_labels', 'gt_ignore_flags'
                                ]),
                            keymap=dict(img='image', gt_bboxes='bboxes')),
                        dict(type='YOLOv5HSVRandomAug'),
                        dict(type='mmdet.RandomFlip', prob=0.5),
                        dict(type='mmdet.PhotoMetricDistortion'),
                        dict(
                            type='mmdet.FilterAnnotations',
                            min_gt_bbox_wh=(8, 8),
                            keep_empty=False),
                        dict(
                            type='mmdet.PackDetInputs',
                            meta_keys=('img_id', 'img_path', 'ori_shape',
                                       'img_shape', 'flip', 'flip_direction'))
                    ])),
            dict(
                type='RepeatDataset',
                times=25,
                dataset=dict(
                    metainfo=dict(
                        classes=('bicycle', 'motorcycle', 'car', 'transporter',
                                 'bus', 'truck', 'trailer', 'unknown',
                                 'mask')),
                    type='YOLOv5CocoDataset',
                    ann_file=
                    '/home/xskalo01/datasets/ndis_park/gt_processed_train.json',
                    data_prefix=dict(img='/home/xskalo01/datasets/ndis_park/'),
                    data_root='/home/xskalo01/datasets/ndis_park/',
                    filter_cfg=dict(filter_empty_gt=False, min_size=32),
                    pipeline=[
                        dict(
                            type='LoadImageFromFile',
                            file_client_args=dict(backend='disk')),
                        dict(type='LoadAnnotations', with_bbox=True),
                        dict(
                            type='mmdet.Resize',
                            scale=(288, 512),
                            keep_ratio=True),
                        dict(
                            type='mmdet.Pad',
                            pad_to_square=False,
                            size=(512, 288),
                            pad_val=dict(img=(114, 114, 114))),
                        dict(
                            type='YOLOv5RandomAffine',
                            scaling_ratio_range=(0.9, 1.5),
                            max_translate_ratio=0.05,
                            max_rotate_degree=5,
                            max_shear_degree=3,
                            border_val=(114, 114, 114)),
                        dict(
                            type='mmdet.CustomCutOut',
                            prob=0.05,
                            cutout_area=(0.05, 0.35),
                            random_pixels=True),
                        dict(
                            type='mmdet.CutOut',
                            n_holes=12,
                            cutout_shape=(20, 20),
                            fill_in=(114, 114, 114)),
                        dict(
                            type='mmdet.Albu',
                            transforms=[
                                dict(type='Blur', p=0.01),
                                dict(type='MedianBlur', p=0.01),
                                dict(type='ToGray', p=0.01),
                                dict(type='CLAHE', p=0.01)
                            ],
                            bbox_params=dict(
                                type='BboxParams',
                                format='pascal_voc',
                                label_fields=[
                                    'gt_bboxes_labels', 'gt_ignore_flags'
                                ]),
                            keymap=dict(img='image', gt_bboxes='bboxes')),
                        dict(type='YOLOv5HSVRandomAug'),
                        dict(type='mmdet.RandomFlip', prob=0.5),
                        dict(type='mmdet.PhotoMetricDistortion'),
                        dict(
                            type='mmdet.FilterAnnotations',
                            min_gt_bbox_wh=(8, 8),
                            keep_empty=False),
                        dict(
                            type='mmdet.PackDetInputs',
                            meta_keys=('img_id', 'img_path', 'ori_shape',
                                       'img_shape', 'flip', 'flip_direction'))
                    ])),
            dict(
                type='RepeatDataset',
                times=5,
                dataset=dict(
                    metainfo=dict(
                        classes=('bicycle', 'motorcycle', 'car', 'transporter',
                                 'bus', 'truck', 'trailer', 'unknown',
                                 'mask')),
                    type='YOLOv5CocoDataset',
                    ann_file=
                    '/home/xskalo01/datasets/MTID/gt_processed_train.json',
                    data_prefix=dict(img='/home/xskalo01/datasets/MTID/'),
                    data_root='/home/xskalo01/datasets/MTID/',
                    filter_cfg=dict(filter_empty_gt=False, min_size=32),
                    pipeline=[
                        dict(
                            type='LoadImageFromFile',
                            file_client_args=dict(backend='disk')),
                        dict(type='LoadAnnotations', with_bbox=True),
                        dict(
                            type='mmdet.Resize',
                            scale=(288, 512),
                            keep_ratio=True),
                        dict(
                            type='mmdet.Pad',
                            pad_to_square=False,
                            size=(512, 288),
                            pad_val=dict(img=(114, 114, 114))),
                        dict(
                            type='YOLOv5RandomAffine',
                            scaling_ratio_range=(0.9, 2),
                            max_translate_ratio=0.05,
                            max_rotate_degree=5,
                            max_shear_degree=3,
                            border_val=(114, 114, 114)),
                        dict(
                            type='mmdet.CustomCutOut',
                            prob=0.05,
                            cutout_area=(0.05, 0.35),
                            random_pixels=True),
                        dict(
                            type='mmdet.CutOut',
                            n_holes=12,
                            cutout_shape=(10, 10),
                            fill_in=(114, 114, 114)),
                        dict(
                            type='mmdet.Albu',
                            transforms=[
                                dict(type='Blur', p=0.01),
                                dict(type='MedianBlur', p=0.01),
                                dict(type='ToGray', p=0.01),
                                dict(type='CLAHE', p=0.01)
                            ],
                            bbox_params=dict(
                                type='BboxParams',
                                format='pascal_voc',
                                label_fields=[
                                    'gt_bboxes_labels', 'gt_ignore_flags'
                                ]),
                            keymap=dict(img='image', gt_bboxes='bboxes')),
                        dict(type='YOLOv5HSVRandomAug'),
                        dict(type='mmdet.RandomFlip', prob=0.5),
                        dict(type='mmdet.PhotoMetricDistortion'),
                        dict(
                            type='mmdet.FilterAnnotations',
                            min_gt_bbox_wh=(8, 8),
                            keep_empty=False),
                        dict(
                            type='mmdet.PackDetInputs',
                            meta_keys=('img_id', 'img_path', 'ori_shape',
                                       'img_shape', 'flip', 'flip_direction'))
                    ])),
            dict(
                type='RepeatDataset',
                times=4,
                dataset=dict(
                    metainfo=dict(
                        classes=('bicycle', 'motorcycle', 'car', 'transporter',
                                 'bus', 'truck', 'trailer', 'unknown',
                                 'mask')),
                    type='YOLOv5CocoDataset',
                    ann_file=
                    '/home/xskalo01/datasets/VisDrone2019-DET-test-dev/gt_processed_train.json',
                    data_prefix=dict(
                        img='/home/xskalo01/datasets/VisDrone2019-DET-test-dev/'
                    ),
                    data_root=
                    '/home/xskalo01/datasets/VisDrone2019-DET-test-dev/',
                    filter_cfg=dict(filter_empty_gt=False, min_size=32),
                    pipeline=[
                        dict(
                            type='LoadImageFromFile',
                            file_client_args=dict(backend='disk')),
                        dict(type='LoadAnnotations', with_bbox=True),
                        dict(
                            type='mmdet.Resize',
                            scale=(288, 512),
                            keep_ratio=True),
                        dict(
                            type='mmdet.Pad',
                            pad_to_square=False,
                            size=(512, 288),
                            pad_val=dict(img=(114, 114, 114))),
                        dict(
                            type='YOLOv5RandomAffine',
                            scaling_ratio_range=(1.5, 2.5),
                            max_translate_ratio=0.05,
                            max_rotate_degree=5,
                            max_shear_degree=3,
                            border_val=(114, 114, 114)),
                        dict(
                            type='mmdet.CustomCutOut',
                            prob=0.05,
                            cutout_area=(0.05, 0.35),
                            random_pixels=True),
                        dict(
                            type='mmdet.CutOut',
                            n_holes=20,
                            cutout_shape=(8, 8),
                            fill_in=(114, 114, 114)),
                        dict(
                            type='mmdet.Albu',
                            transforms=[
                                dict(type='Blur', p=0.01),
                                dict(type='MedianBlur', p=0.01),
                                dict(type='ToGray', p=0.01),
                                dict(type='CLAHE', p=0.01)
                            ],
                            bbox_params=dict(
                                type='BboxParams',
                                format='pascal_voc',
                                label_fields=[
                                    'gt_bboxes_labels', 'gt_ignore_flags'
                                ]),
                            keymap=dict(img='image', gt_bboxes='bboxes')),
                        dict(type='YOLOv5HSVRandomAug'),
                        dict(type='mmdet.RandomFlip', prob=0.5),
                        dict(type='mmdet.PhotoMetricDistortion'),
                        dict(
                            type='mmdet.FilterAnnotations',
                            min_gt_bbox_wh=(8, 8),
                            keep_empty=False),
                        dict(
                            type='mmdet.PackDetInputs',
                            meta_keys=('img_id', 'img_path', 'ori_shape',
                                       'img_shape', 'flip', 'flip_direction'))
                    ])),
            dict(
                type='RepeatDataset',
                times=1,
                dataset=dict(
                    metainfo=dict(
                        classes=('bicycle', 'motorcycle', 'car', 'transporter',
                                 'bus', 'truck', 'trailer', 'unknown',
                                 'mask')),
                    type='YOLOv5CocoDataset',
                    ann_file=
                    '/home/xskalo01/datasets/DETRAC/gt_processed_train.json',
                    data_prefix=dict(img='/home/xskalo01/datasets/DETRAC/'),
                    data_root='/home/xskalo01/datasets/DETRAC/',
                    filter_cfg=dict(filter_empty_gt=False, min_size=32),
                    pipeline=[
                        dict(
                            type='LoadImageFromFile',
                            file_client_args=dict(backend='disk')),
                        dict(type='LoadAnnotations', with_bbox=True),
                        dict(
                            type='mmdet.Resize',
                            scale=(288, 512),
                            keep_ratio=True),
                        dict(
                            type='mmdet.Pad',
                            pad_to_square=False,
                            size=(512, 288),
                            pad_val=dict(img=(114, 114, 114))),
                        dict(
                            type='YOLOv5RandomAffine',
                            scaling_ratio_range=(0.8, 1),
                            max_translate_ratio=0.05,
                            max_rotate_degree=5,
                            max_shear_degree=3,
                            border_val=(114, 114, 114)),
                        dict(
                            type='mmdet.CustomCutOut',
                            prob=0.05,
                            cutout_area=(0.05, 0.35),
                            random_pixels=True),
                        dict(
                            type='mmdet.CutOut',
                            n_holes=6,
                            cutout_shape=(22, 22),
                            fill_in=(114, 114, 114)),
                        dict(
                            type='mmdet.Albu',
                            transforms=[
                                dict(type='Blur', p=0.01),
                                dict(type='MedianBlur', p=0.01),
                                dict(type='ToGray', p=0.01),
                                dict(type='CLAHE', p=0.01)
                            ],
                            bbox_params=dict(
                                type='BboxParams',
                                format='pascal_voc',
                                label_fields=[
                                    'gt_bboxes_labels', 'gt_ignore_flags'
                                ]),
                            keymap=dict(img='image', gt_bboxes='bboxes')),
                        dict(type='YOLOv5HSVRandomAug'),
                        dict(type='mmdet.RandomFlip', prob=0.5),
                        dict(type='mmdet.PhotoMetricDistortion'),
                        dict(
                            type='mmdet.FilterAnnotations',
                            min_gt_bbox_wh=(8, 8),
                            keep_empty=False),
                        dict(
                            type='mmdet.PackDetInputs',
                            meta_keys=('img_id', 'img_path', 'ori_shape',
                                       'img_shape', 'flip', 'flip_direction'))
                    ]))
        ]))
val_pipeline = [
    dict(type='LoadImageFromFile', file_client_args=dict(backend='disk')),
    dict(type='LoadAnnotations', with_bbox=True, _scope_='mmdet'),
    dict(
        type='LetterResize',
        scale=(512, 288),
        allow_scale_up=False,
        pad_val=dict(img=114)),
    dict(
        type='mmdet.PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor', 'pad_param'))
]
test_pipeline = [
    dict(type='LoadImageFromFile', file_client_args=dict(backend='disk')),
    dict(type='LoadAnnotations', with_bbox=True, _scope_='mmdet'),
    dict(
        type='LetterResize',
        scale=(512, 288),
        allow_scale_up=False,
        pad_val=dict(img=114)),
    dict(
        type='mmdet.PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor', 'pad_param'))
]
val_dataloader = dict(
    batch_size=1,
    num_workers=16,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        metainfo=dict(
            classes=('bicycle', 'motorcycle', 'car', 'transporter', 'bus',
                     'truck', 'trailer', 'unknown', 'mask')),
        type='YOLOv5CocoDataset',
        data_root='/home/xskalo01/datasets/',
        ann_file='val.json',
        data_prefix=dict(img=''),
        test_mode=True,
        pipeline=[
            dict(
                type='LoadImageFromFile',
                file_client_args=dict(backend='disk')),
            dict(type='LoadAnnotations', with_bbox=True, _scope_='mmdet'),
            dict(
                type='LetterResize',
                scale=(512, 288),
                allow_scale_up=False,
                pad_val=dict(img=114)),
            dict(
                type='mmdet.PackDetInputs',
                meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                           'scale_factor', 'pad_param'))
        ]))
test_dataloader = dict(
    batch_size=1,
    num_workers=16,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        metainfo=dict(
            classes=('bicycle', 'motorcycle', 'car', 'transporter', 'bus',
                     'truck', 'trailer', 'unknown', 'mask')),
        type='YOLOv5CocoDataset',
        data_root='/home/xskalo01/datasets/',
        ann_file='test.json',
        data_prefix=dict(img=''),
        test_mode=True,
        pipeline=[
            dict(
                type='LoadImageFromFile',
                file_client_args=dict(backend='disk')),
            dict(type='LoadAnnotations', with_bbox=True, _scope_='mmdet'),
            dict(
                type='LetterResize',
                scale=(512, 288),
                allow_scale_up=False,
                pad_val=dict(img=114)),
            dict(
                type='mmdet.PackDetInputs',
                meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                           'scale_factor', 'pad_param'))
        ]))
val_evaluator = dict(
    type='mmdet.CocoMetric',
    proposal_nums=(100, 1, 10),
    ann_file='/home/xskalo01/datasets/val.json',
    metric='bbox')
test_evaluator = dict(
    type='mmdet.CocoMetric',
    proposal_nums=(100, 1, 10),
    ann_file='/home/xskalo01/datasets/test.json',
    metric='bbox')
optim_wrapper = dict(
    type='OptimWrapper',
    clip_grad=dict(max_norm=10.0),
    optimizer=dict(
        type='SGD',
        lr=base_lr,
        momentum=0.937,
        weight_decay=0.0005,
        nesterov=True,
        batch_size_per_gpu=192),
    constructor='YOLOv5OptimizerConstructor')
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50),
    param_scheduler=dict(
        type='YOLOv5ParamSchedulerHook',
        scheduler_type='linear',
        lr_factor=0.01,
        max_epochs=max_epochs,
        warmup_epochs=5),
    checkpoint=dict(
        type='CheckpointHook',
        interval=1,
        save_best='auto',
        max_keep_ckpts=100),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='mmdet.DetVisualizationHook'))
custom_hooks = [
    dict(
        type='EMAHook',
        ema_type='ExpMomentumEMA',
        momentum=0.0001,
        update_buffers=True,
        strict_load=False,
        priority=49),
    dict(
        type='mmdet.PipelineSwitchHook',
        switch_epoch=290,
        switch_pipeline=[
            dict(
                type='LoadImageFromFile',
                file_client_args=dict(backend='disk')),
            dict(type='LoadAnnotations', with_bbox=True),
            dict(type='mmdet.Resize', scale=(288, 512), keep_ratio=True),
            dict(
                type='mmdet.Pad',
                pad_to_square=False,
                size=(512, 288),
                pad_val=dict(img=(114, 114, 114))),
            dict(
                type='YOLOv5RandomAffine',
                max_rotate_degree=0,
                max_shear_degree=0,
                scaling_ratio_range=(0.9, 1.2),
                max_aspect_ratio=100,
                border_val=(114, 114, 114)),
            dict(
                type='mmdet.Albu',
                transforms=[
                    dict(type='Blur', p=0.01),
                    dict(type='MedianBlur', p=0.01),
                    dict(type='ToGray', p=0.01),
                    dict(type='CLAHE', p=0.01)
                ],
                bbox_params=dict(
                    type='BboxParams',
                    format='pascal_voc',
                    label_fields=['gt_bboxes_labels', 'gt_ignore_flags']),
                keymap=dict(img='image', gt_bboxes='bboxes')),
            dict(type='YOLOv5HSVRandomAug'),
            dict(type='mmdet.RandomFlip', prob=0.5),
            dict(type='mmdet.PhotoMetricDistortion'),
            dict(
                type='mmdet.FilterAnnotations',
                min_gt_bbox_wh=(8, 8),
                keep_empty=False),
            dict(
                type='mmdet.PackDetInputs',
                meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                           'flip', 'flip_direction'))
        ])
]
train_cfg = dict(
    type='EpochBasedTrainLoop',
    max_epochs=max_epochs,
    val_interval=1,
    dynamic_intervals=[(290, 1)])
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')
log_level = 'INFO'
log_processor = dict(type='LogProcessor', window_size=50, by_epoch=True)
visualizer = dict(
    type='mmdet.DetLocalVisualizer',
    vis_backends=[
        dict(type='LocalVisBackend'),
        dict(type='TensorboardVisBackend')
    ],
    name='visualizer',
    line_width=2,
    alpha=0.9)
env_cfg = dict(
    cudnn_benchmark=True,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'))
last_stage_out_channels = 768
strides = [8, 16, 32]


backbone_pruning_ratio = 0.6
bbox_head_pruning_ratio = 0.7
model = dict(
    _scope_='mmrazor',
    type='ItePruneAlgorithm',
    architecture=dict(
        type='YOLODetector',
        data_preprocessor=dict(
            type='YOLOv5DetDataPreprocessor',
            mean=[0.0, 0.0, 0.0],
            std=[255.0, 255.0, 255.0],
            bgr_to_rgb=True),
        backbone=dict(
            type='YOLOv8CSPDarknet',
            arch='P5',
            last_stage_out_channels=768,
            deepen_factor=0.33,
            widen_factor=0.25,
            norm_cfg=dict(type='BN', momentum=0.03, eps=0.001),
            act_cfg=dict(type='SiLU', inplace=True)),
        neck=dict(
            type='YOLOv8PAFPN',
            deepen_factor=0.33,
            widen_factor=0.25,
            in_channels=[256, 512, 768],
            out_channels=[256, 512, 768],
            num_csp_blocks=3,
            norm_cfg=dict(type='BN', momentum=0.03, eps=0.001),
            act_cfg=dict(type='SiLU', inplace=True)),
        bbox_head=dict(
            type='YOLOv8Head',
            head_module=dict(
                type='YOLOv8HeadModule',
                num_classes=9,
                in_channels=[256, 512, 768],
                widen_factor=0.25,
                reg_max=16,
                norm_cfg=dict(type='BN', momentum=0.03, eps=0.001),
                act_cfg=dict(type='SiLU', inplace=True),
                featmap_strides=[8, 16, 32]),
            prior_generator=dict(
                type='mmdet.MlvlPointGenerator',
                offset=0.5,
                strides=[8, 16, 32]),
            bbox_coder=dict(type='DistancePointBBoxCoder'),
            loss_cls=dict(
                type='mmdet.CrossEntropyLoss',
                use_sigmoid=True,
                reduction='none',
                loss_weight=0.5),
            loss_bbox=dict(
                type='IoULoss',
                iou_mode='ciou',
                bbox_format='xyxy',
                reduction='sum',
                loss_weight=7.5,
                return_iou=False),
            loss_dfl=dict(
                type='mmdet.DistributionFocalLoss',
                reduction='mean',
                loss_weight=0.375)),
        train_cfg=dict(
            assigner=dict(
                type='BatchTaskAlignedAssigner',
                num_classes=9,
                use_ciou=True,
                topk=10,
                alpha=0.5,
                beta=6.0,
                eps=1e-09)),
        test_cfg=dict(
            multi_label=True,
            nms_pre=30000,
            score_thr=0.001,
            nms=dict(type='nms', iou_threshold=0.7),
            max_per_img=300),
        _scope_='mmyolo',
        init_cfg=dict(
            type='Pretrained',
            checkpoint=
            '/home/xskalo01/bp/proj/working_dir_yolov8_n_conf8_512x288/epoch_300.pth'
        )),
    target_pruning_ratio=dict({
        'backbone.stem.conv_(0, 16)_16':
        backbone_pruning_ratio,
        'backbone.stage4.2.conv1.conv_(0, 96)_96':
        backbone_pruning_ratio,
        'bbox_head.head_module.cls_preds.2.0.conv_(0, 64)_64':
        bbox_head_pruning_ratio,
        'bbox_head.head_module.cls_preds.2.1.conv_(0, 64)_64':
        bbox_head_pruning_ratio,
        'bbox_head.head_module.reg_preds.2.0.conv_(0, 64)_64':
        bbox_head_pruning_ratio,
        'bbox_head.head_module.reg_preds.2.1.conv_(0, 64)_64':
        bbox_head_pruning_ratio,
        'bbox_head.head_module.cls_preds.0.0.conv_(0, 64)_64':
        bbox_head_pruning_ratio,
        'bbox_head.head_module.cls_preds.0.1.conv_(0, 64)_64':
        bbox_head_pruning_ratio,
        'bbox_head.head_module.reg_preds.0.0.conv_(0, 64)_64':
        bbox_head_pruning_ratio,
        'bbox_head.head_module.reg_preds.0.1.conv_(0, 64)_64':
        bbox_head_pruning_ratio,
        'bbox_head.head_module.cls_preds.1.0.conv_(0, 64)_64':
        bbox_head_pruning_ratio,
        'bbox_head.head_module.cls_preds.1.1.conv_(0, 64)_64':
        bbox_head_pruning_ratio,
        'bbox_head.head_module.reg_preds.1.0.conv_(0, 64)_64':
        bbox_head_pruning_ratio,
        'bbox_head.head_module.reg_preds.1.1.conv_(0, 64)_64':
        bbox_head_pruning_ratio,
    }),
    mutator_cfg=dict(
        type='ChannelMutator',
        channel_unit_cfg=dict(
            type='L1MutableChannelUnit',
            default_args=dict(choice_mode='ratio')),
        parse_cfg=dict(
            type='ChannelAnalyzer',
            tracer_type='FxTracer',
            demo_input=dict(type='DefaultDemoInput', scope='mmyolo'))))
import sys; sys.path.append(repo_path)
