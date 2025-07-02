from .esdnet import ESDNet
from .unet import UNet
from .unet3plus import UNet3Plus
from .myunet import MyUNet
from .myunet3plus import MyUNet3Plus
from .mysegformer import MySegFormer
from .myuformer import MyUFormer

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
            num_features=args.FEAT_DIM,
        )
    elif model_name == 'myunet':
        model = MyUNet(
            num_features=args.FEAT_DIM,
        )
    elif model_name == 'myunet3plus':
        model = MyUNet3Plus(
        )

    elif model_name == 'mysegformer':
        model = MySegFormer(
            model_name=args.ENCODER_NAME,
        )
    elif model_name == 'myuformer':
        model = MyUFormer(
            arch=args.ARCH,
        )
    else:
        raise NotImplementedError(f"Unknown model: {model_name}")
    return model
