_base_ = ['./base_static.py']
h = 384
w = 640
onnx_config = dict(input_shape=(w, h))
backend_config = dict(
    type='tensorrt',
    common_config=dict(fp16_mode=False, max_workspace_size=1 << 30),
    model_inputs=[
        dict(
            input_shapes=dict(
                input=dict(
                    min_shape=[1, 3, w, h],
                    opt_shape=[1, 3, w, h],
                    max_shape=[1, 3, w, h])))
    ])
use_efficientnms = False  # whether to replace TRTBatchedNMS plugin with EfficientNMS plugin # noqa E501
