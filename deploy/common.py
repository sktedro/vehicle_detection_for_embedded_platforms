import mmengine

def get_deploy_config(deploy_cfg_filepath: str, model_cfg: mmengine.Config, deployed_filename: str):
    with open(deploy_cfg_filepath) as f:
        deploy_cfg_str = f.read()
    w, h = model_cfg[0]["img_scale"]
    deploy_cfg_str = deploy_cfg_str.replace('"WIDTH_PLACEHOLDER"', str(w))
    deploy_cfg_str = deploy_cfg_str.replace('"HEIGHT_PLACEHOLDER"', str(h))
    deploy_cfg_str = deploy_cfg_str.replace('FILENAME_PLACEHOLDER', deployed_filename)
    return mmengine.Config.fromstring(deploy_cfg_str, ".py")