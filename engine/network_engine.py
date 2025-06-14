from model.model import network

def CPDGL(args):

    model = network(args,
    backbone=args.backbone,
    num_classes=args.num_classes,
    pretrained=args.pretrained,
    init_momentum=args.momentum,
    aux_layer=args.aux_layer
    )
    param_groups = model.get_param_groups()

    return model, param_groups