import cv_lib.augmentation as aug


imagenet_val_aug = aug.Compose(
    aug.Resize(256),
    aug.CenterCrop((224, 224))
)


__REGISTERED_AUG__ = {
    "mnist_train": None,
    "cifar_10_train": None,
    "cifar_100_train": None,
    "imagenet_train": imagenet_val_aug,
}


def get_data_aug(dataset_name: str, split: str):
    if "mnist" in dataset_name.lower():
        dataset_name = "mnist"
    name = "{}_{}".format(dataset_name, split)
    return __REGISTERED_AUG__[name]

