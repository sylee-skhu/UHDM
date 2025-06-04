from .esdnet import ESDNet
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
    elif model_name == 'mynet':
        model = MyNet(
            num_channels=args.EN_FEATURE_NUM,
            num_features=args.EN_FEATURE_NUM,
        )
    else:
        raise NotImplementedError(f"Unknown model: {model_name}")
    return model
