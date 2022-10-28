from .focal import small_focal

def focal_small_224(args):
    model = small_focal(
                num_classes = args.num_classes,
                drop_path_rate = args.drop_path
                )
    return model