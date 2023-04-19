backend_config = dict(type='torchscript')
ir_config = dict(
    type='torchscript',
    save_file='end2end.pt',
    input_names=['input'],
    output_names=['dets', 'labels'],
    input_shape=None)
codebase_config = dict(
    type='mmdet',
    task='ObjectDetection',
    model_type='end2end',
    post_processing=dict(
        score_threshold=0.05,
        confidence_threshold=0.005,
        iou_threshold=0.5,
        max_output_boxes_per_class=200,
        pre_top_k=5000,
        keep_top_k=100,
        background_label_id=-1,
    ))

