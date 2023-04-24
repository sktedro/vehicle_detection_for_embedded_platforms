# This will be replaced before deploying, by common.deploy_config_from_file()
w = "WIDTH_PLACEHOLDER"
h = "HEIGHT_PLACEHOLDER"
output_filename = "FILENAME_PLACEHOLDER"

onnx_config = dict(
    type='onnx',
    export_params=True,
    keep_initializers_as_inputs=False,
    opset_version=11,
    save_file=output_filename + '.onnx',
    input_names=['input'],
    output_names=['dets', 'labels'],
    input_shape=(w, h), # Shouldn't this be reversed? No (tested)
    optimize=True)

codebase_config = dict(
    type='mmyolo',
    task='ObjectDetection',
    model_type='ncnn_end2end',
    post_processing=dict(
        score_threshold=0.05,
        confidence_threshold=0.005,
        iou_threshold=0.5,
        max_output_boxes_per_class=200,
        pre_top_k=5000,
        keep_top_k=100,
        background_label_id=-1),
    module=['mmyolo.deploy'])

backend_config = dict(
    type='ncnn',
    precision='FP32',
    use_vulkan=False)
