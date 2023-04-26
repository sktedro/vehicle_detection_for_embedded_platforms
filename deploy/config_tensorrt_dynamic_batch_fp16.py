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
    input_shape=(w, h), # w*h (should be h*w according to MMDeploy docs, but that doesn't work!)
    dynamic_axes={ # Use a dynamic model, but only to be able to use batches
        'input': {
            0: 'batch',
        },
        'dets': {
            0: 'batch',
            1: 'num_dets',
        },
        'labels': {
            0: 'batch',
            1: 'num_dets',
        },
    },
    optimize=True)

codebase_config = dict(
    type='mmyolo',
    task='ObjectDetection',
    model_type='end2end',
    post_processing=dict(
        iou_threshold=0.5, # Only used if not specified in model config, but it needs to be specified there, so this line is not used
        score_threshold=0.05, # Only used if not specified in model config, but it needs to be specified there, so this line is not used
        confidence_threshold=0.005,
        max_output_boxes_per_class=200,
        pre_top_k=5000,
        keep_top_k=100,
        background_label_id=-1),
    module=['mmyolo.deploy'])

backend_config = dict(
    type='tensorrt',
    common_config=dict(
        fp16_mode=True,
        int8_mode=False,
        max_workspace_size=1 << 30
        ),
    model_inputs=[
        dict(
            input_shapes=dict(
                input=dict(
                    min_shape=[1, 3, h, w],
                    opt_shape=[1, 3, h, w],
                    max_shape=[32, 3, h, w])))])

use_efficientnms = False # EfficientNMS doesn't work for some reason
