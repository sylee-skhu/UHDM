import os
import torch
from thop import profile, clever_format
from models import create_model
from config.config import get_parser

def measure_flops_and_params(model_cfg, img_size=256, device=None):
    # set up device
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    # create and load model
    model = create_model(model_cfg).to(device)
    model.eval()

    # (optional) load checkpoint if your config points to one
    # ckpt = torch.load(model_cfg.LOAD_PATH, map_location=device)
    # state_dict = ckpt.get('state_dict', ckpt)
    # state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    # model.load_state_dict(state_dict)

    # prepare a dummy input dict matching your forward signature
    input_tensor = torch.randn(1, 3, img_size, img_size, device=device)
    input_dict = {'in_img': input_tensor, 'label': input_tensor}

    # run thop.profile
    macs, params = profile(
        model,
        inputs=(input_dict,),
        verbose=True,
    )

    print(model.flops())
    macs, params = clever_format([macs, params], "%.3f")
    print("MACs:" + macs + ", Params:" + params)
    

if __name__ == "__main__":
    args = get_parser()
    measure_flops_and_params(args)
