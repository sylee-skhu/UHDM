from .vggperceptual import multi_VGGPerceptualLoss
from .myloss import MyLoss


def create_loss(args):
    loss_name = args.LOSS_NAME.lower()
    if loss_name == 'vggperceptual':
        loss = multi_VGGPerceptualLoss(
            lam=args.LAM,
            lam_p=args.LAM_P
        )
    elif loss_name == 'myloss':
        loss = MyLoss(
            lam_p=args.LAM_P,
            lam_adv=args.LAM_ADV
        )
    else:
        raise NotImplementedError(f"Unknown loss: {loss_name}")
    return loss
