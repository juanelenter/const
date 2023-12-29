import os
import shutil
import subprocess
import time
from types import SimpleNamespace

import torchvision as tv
import torchvision.transforms as tv_tforms

import shared

logger = shared.fetch_main_logger()


def load_mnist_dataset(data_path):
    logger.info(f"Loading MNIST dataset from {data_path}")
    num_classes = 10
    train_transforms = tv_tforms.Compose([tv_tforms.ToTensor(), tv_tforms.Normalize((0.1307,), (0.3081,))])
    train_dataset = tv.datasets.MNIST(root=data_path, train=True, transform=train_transforms, download=False)
    val_dataset = tv.datasets.MNIST(root=data_path, train=False, transform=train_transforms, download=False)
    return SimpleNamespace(train=train_dataset, val=val_dataset), num_classes


def load_cifar10_dataset(data_path, use_data_augmentation=True):
    logger.info(f"Loading CIFAR dataset from {data_path}")
    num_classes = 10

    default_transforms = [tv_tforms.ToTensor(), tv_tforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))]

    if use_data_augmentation:
        augmentation_transforms = [tv_tforms.RandomCrop(32, padding=4), tv_tforms.RandomHorizontalFlip()]
        train_transforms = augmentation_transforms + default_transforms
    else:
        train_transforms = default_transforms

    train_transforms = tv_tforms.Compose(train_transforms)
    val_transforms = tv_tforms.Compose(default_transforms)

    train_dataset = tv.datasets.CIFAR10(root=data_path, train=True, transform=train_transforms, download=True)
    val_dataset = tv.datasets.CIFAR10(root=data_path, train=False, transform=val_transforms, download=True)

    return SimpleNamespace(train=train_dataset, val=val_dataset), num_classes


def load_imagenet_dataset(data_path, use_data_augmentation=True):
    if data_path is None:
        data_path = os.path.join(os.environ["SLURM_TMPDIR"], "imagenet")

    logger.info(f"Loading ImageNet dataset from {data_path}")
    num_classes = 1000

    # Transforms match ImageNet training for ResNet50-V1 and ResNet18-V1)
    # https://pytorch.org/vision/stable/models/generated/torchvision.models.resnet50.html#torchvision.models.ResNet50_Weights

    default_transforms = [tv_tforms.ToTensor(), tv_tforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]
    interpolation = tv_tforms.functional.InterpolationMode.BILINEAR

    if use_data_augmentation:
        augmentation_transforms = [
            tv_tforms.RandomResizedCrop(224, interpolation=interpolation),
            tv_tforms.RandomHorizontalFlip(0.5),
        ]
        train_transforms = augmentation_transforms + default_transforms
    else:
        train_transforms = default_transforms

    train_transforms = tv_tforms.Compose(train_transforms)
    val_transforms = tv_tforms.Compose(
        [tv_tforms.Resize(256, interpolation=interpolation), tv_tforms.CenterCrop(224)] + default_transforms
    )

    train_dataset = tv.datasets.ImageNet(root=data_path, split="train", transform=train_transforms)
    val_dataset = tv.datasets.ImageNet(root=data_path, split="val", transform=val_transforms)
    return SimpleNamespace(train=train_dataset, val=val_dataset), num_classes


def copy_imagenet_to_tmpdir(src_dir):
    copy_start = time.time()

    TMPDIR_PATH = os.path.join(os.environ["SLURM_TMPDIR"], "imagenet")
    IMAGENET_TORCHVISION_DIR = "/network/datasets/imagenet.var/imagenet_torchvision"

    logger.info(f"Copying Imagenet data to SLURM_TMPDIR: {TMPDIR_PATH}")
    if os.path.exists(TMPDIR_PATH):
        logger.info("SLURM_TMPDIR already contains a folder called `imagenet`, skipping copy")
        return

    os.makedirs(TMPDIR_PATH, exist_ok=True)
    subprocess.run(["chmod", "+rwx", TMPDIR_PATH])

    logger.info("Copying validation data")
    shutil.copytree(
        f"{IMAGENET_TORCHVISION_DIR}/val", os.path.join(os.environ["SLURM_TMPDIR"], "imagenet"), dirs_exist_ok=True
    )

    for file_name in ["ILSVRC2012_img_val.tar", "ILSVRC2012_devkit_t12.tar.gz"]:
        if not os.path.exists(os.path.join(TMPDIR_PATH, file_name)):
            logger.info(f"Copying {file_name} to {TMPDIR_PATH}")
            shutil.copy(f"/network/datasets/imagenet/{file_name}", os.path.join(TMPDIR_PATH, file_name))

    logger.info("Copying training data")
    if not os.path.exists(os.path.join(TMPDIR_PATH, "train")):
        os.makedirs(os.path.join(TMPDIR_PATH, "train"), exist_ok=True)
        subprocess.run(["chmod", "+rwx", os.path.join(TMPDIR_PATH, "train")])
        os.chdir(os.path.join(TMPDIR_PATH, "train"))
        tar_command = "tar -xf /network/datasets/imagenet/ILSVRC2012_img_train.tar --to-command='mkdir ${TAR_REALNAME%.tar}; tar -xC ${TAR_REALNAME%.tar}'"
        subprocess.run(tar_command, shell=True)

    logger.info("Copying auxiliary files")
    os.makedirs(os.path.join(TMPDIR_PATH, "scripts"), exist_ok=True)
    subprocess.run(["chmod", "+rwx", os.path.join(TMPDIR_PATH, "scripts")])
    shutil.copytree(f"{IMAGENET_TORCHVISION_DIR}/scripts", os.path.join(TMPDIR_PATH, "scripts"), dirs_exist_ok=True)
    shutil.copy(f"{IMAGENET_TORCHVISION_DIR}/meta.bin", os.path.join(TMPDIR_PATH, "meta.bin"))

    logger.info(f"Contents inside {TMPDIR_PATH}:")
    logger.info(os.listdir(TMPDIR_PATH))

    copy_end = time.time()
    logger.info("Completed copying of ImageNet dataset")
    logger.info(f"Total elapsed copy and extract time {copy_end - copy_start} seconds")

    # cd back to original directory
    os.chdir(src_dir)


def copy_imagenet_debug_to_tmpdir(src_dir):
    copy_start = time.time()

    TMP_DIR, DATA_DIR = os.environ["SLURM_TMPDIR"], os.environ["DATA_DIR"]
    logger.info(f"Copying Imagenet data to SLURM_TMPDIR: {TMP_DIR}")

    if os.path.exists(os.path.join(TMP_DIR, "imagenet")):
        logger.info("SLURM_TMPDIR already contains a folder called `imagenet`, skipping copy")
        return

    rsync_command = f"rsync -a {DATA_DIR}/imagenet_debug.tar.gz {TMP_DIR}/"
    subprocess.run(rsync_command, shell=True)

    tar_command = f"tar --skip-old-files -xzf {TMP_DIR}/imagenet_debug.tar.gz -C {TMP_DIR}"
    subprocess.run(tar_command, shell=True)

    logger.info(f"Contents inside {os.path.join(TMP_DIR, 'imagenet')}:")
    logger.info(os.listdir(os.path.join(TMP_DIR, "imagenet")))

    copy_end = time.time()
    logger.info("Completed copying of *DEBUG* ImageNet dataset")
    logger.info(f"Total elapsed copy and extract time {copy_end - copy_start} seconds")

    # cd back to original directory
    os.chdir(src_dir)
