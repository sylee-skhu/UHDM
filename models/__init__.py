from .esdnet import ESDNet
from .unet import UNet
from .mynet import MyNet


def create_model(args):
    model_name = args.MODEL_NAME.lower()
    if model_name == 'esdnet':
        model = ESDNet(
            en_feature_num=args.EN_FEATURE_NUM,
            en_inter_num=args.EN_INTER_NUM,
            de_feature_num=args.DE_FEATURE_NUM,
            de_inter_num=args.DE_INTER_NUM,
            sam_number=args.SAM_NUMBER,
        )
    elif model_name == 'unet':
        model = UNet(
            num_channels=args.IN_DIM,
            num_features=args.FEAT_DIM,
            num_out_channels=args.OUT_DIM,
        )
    elif model_name == 'mynet':
        model = MyNet(
            num_channels=args.CH_DIM,
            num_features=args.FEAT_DIM,
        )
    else:
        raise NotImplementedError(f"Unknown model: {model_name}")
    return model
