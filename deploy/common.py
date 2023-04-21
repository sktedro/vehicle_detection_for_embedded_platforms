import mmengine

def get_deploy_config(deploy_cfg_filepath: str, model_cfg: mmengine.Config, deployed_filename):
    with open(deploy_cfg_filepath) as f:
        deploy_cfg_str = f.read()
    deploy_cfg_str = deploy_cfg_str.replace('"HEIGHT_PLACEHOLDER"', str(model_cfg[0]["img_scale"][0]))
    deploy_cfg_str = deploy_cfg_str.replace('"WIDTH_PLACEHOLDER"', str(model_cfg[0]["img_scale"][1]))
    deploy_cfg_str = deploy_cfg_str.replace('FILENAME_PLACEHOLDER', deployed_filename)
    return mmengine.Config.fromstring(deploy_cfg_str, ".py")