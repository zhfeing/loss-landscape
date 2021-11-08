import cv_lib.augmentation as aug


mnist_train_aug = aug.Compose(
    aug.RandomRotation((-30, 30))
)

cifar_train_aug = aug.Compose(
    aug.RandomCrop((32, 32), padding=4),
    aug.RandomHorizontalFlip()
)

imagenet_train_aug = aug.Compose(
    aug.RandomResizedCrop(size=(224, 224), scale=(0.6, 1)),
    aug.RandomHorizontalFlip()
)
imagenet_val_aug = aug.Compose(
    aug.Resize(256),
    aug.CenterCrop((224, 224))
)


__REGISTERED_AUG__ = {
    "mnist_train": mnist_train_aug,
    "mnist_val": None,
    "cifar_10_train": cifar_train_aug,
    "cifar_10_val": None,
    "cifar_100_train": cifar_train_aug,
    "cifar_100_val": None,
    "imagenet_train": imagenet_train_aug,
    "imagenet_val": imagenet_val_aug
}


def get_data_aug(dataset_name: str, split: str):
    if "mnist" in dataset_name.lower():
        dataset_name = "mnist"
    name = "{}_{}".format(dataset_name, split)
    return __REGISTERED_AUG__[name]

