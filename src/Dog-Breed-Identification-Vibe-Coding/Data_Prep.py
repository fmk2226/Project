"""数据读取、分层切分和图像预处理。"""

from __future__ import annotations

import csv
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models import ResNet34_Weights


LabelRow = tuple[str, int]
SMOKE_SAMPLE_LIMIT = 1200
VALIDATION_RESIZE_RATIO = 0.875
PREFETCH_FACTOR = 3


class DogDataset(Dataset):
    def __init__(self, image_dir: Path, rows: list[LabelRow], transform):
        self.image_dir = Path(image_dir)
        self.rows = rows
        self.transform = transform

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        image_id, label = self.rows[index]
        with Image.open(self.image_dir / f"{image_id}.jpg") as image:
            image = image.convert("RGB")
            return self.transform(image), label


class TestDogDataset(Dataset):
    def __init__(self, image_dir: Path, transform):
        self.image_paths = sorted(Path(image_dir).glob("*.jpg"))
        if not self.image_paths:
            raise FileNotFoundError(f"no test jpg images found under {image_dir}")
        self.transform = transform

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, index: int):
        image_path = self.image_paths[index]
        with Image.open(image_path) as image:
            image = image.convert("RGB")
            return self.transform(image), image_path.stem


def load_data(base_dir: Path, demo: bool = False) -> Path:
    """保持原项目入口风格，返回原始数据目录。"""
    folder = "kaggle_dog_tiny" if demo else "dog-breed-identification"
    data_dir = Path(base_dir) / "data" / folder
    if not data_dir.exists():
        raise FileNotFoundError(f"data directory does not exist: {data_dir}")
    return data_dir


def read_csv_labels(fname: Path) -> dict[str, str]:
    with Path(fname).open(newline="", encoding="utf-8") as file:
        return {row["id"]: row["breed"] for row in csv.DictReader(file)}


def _seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def _read_labeled_samples(
    data_dir: Path,
) -> tuple[list[str], list[int], list[str]]:
    labels = read_csv_labels(data_dir / "labels.csv")
    classes = sorted(set(labels.values()))
    class_to_idx = {name: index for index, name in enumerate(classes)}
    image_ids = list(labels)
    targets = [class_to_idx[labels[image_id]] for image_id in image_ids]
    return image_ids, targets, classes


def _limit_smoke_samples(
    image_ids: list[str],
    targets: list[int],
    seed: int,
) -> tuple[list[str], list[int]]:
    image_ids, _, targets, _ = train_test_split(
        image_ids,
        targets,
        train_size=min(SMOKE_SAMPLE_LIMIT, len(image_ids)),
        random_state=seed,
        stratify=targets,
    )
    return image_ids, targets


def _split_rows(
    data_dir: Path,
    split_seed: int,
    val_ratio: float,
    smoke: bool,
) -> tuple[list[LabelRow], list[LabelRow], list[str]]:
    image_ids, targets, classes = _read_labeled_samples(Path(data_dir))

    if smoke:
        image_ids, targets = _limit_smoke_samples(
            image_ids,
            targets,
            split_seed,
        )

    train_ids, val_ids, train_targets, val_targets = train_test_split(
        image_ids,
        targets,
        test_size=val_ratio,
        random_state=split_seed,
        stratify=targets,
    )
    return (
        list(zip(train_ids, train_targets)),
        list(zip(val_ids, val_targets)),
        classes,
    )


def build_transforms(
    image_size: int,
    augment: str,
) -> tuple[transforms.Compose, transforms.Compose]:
    weights_transform = ResNet34_Weights.DEFAULT.transforms()
    normalize = transforms.Normalize(weights_transform.mean, weights_transform.std)
    resized_size = round(image_size / VALIDATION_RESIZE_RATIO)

    if augment == "vit":
        # 复现 Kaggle 高票 ViT-L 方案：尽量保留完整犬只，只做温和几何扰动。
        train_transform = transforms.Compose(
            [
                transforms.Resize(resized_size),
                transforms.CenterCrop(image_size),
                transforms.RandomHorizontalFlip(p=0.6),
                transforms.RandomRotation(degrees=30),
                transforms.ToTensor(),
                normalize,
            ]
        )
    elif augment == "original":
        train_transform = transforms.Compose(
            [
                transforms.RandomResizedCrop(
                    image_size,
                    scale=(0.08, 1.0),
                    ratio=(0.75, 1.333),
                ),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(
                    brightness=0.4,
                    contrast=0.4,
                    saturation=0.4,
                ),
                transforms.ToTensor(),
                normalize,
            ]
        )
    elif augment == "strong":
        train_transform = transforms.Compose(
            [
                transforms.RandomResizedCrop(
                    image_size,
                    scale=(0.55, 1.0),
                    ratio=(0.75, 1.333),
                ),
                transforms.RandomHorizontalFlip(),
                transforms.RandAugment(num_ops=2, magnitude=7),
                transforms.ToTensor(),
                normalize,
                transforms.RandomErasing(
                    p=0.20,
                    scale=(0.02, 0.15),
                    ratio=(0.3, 3.3),
                ),
            ]
        )
    else:
        train_transform = transforms.Compose(
            [
                transforms.RandomResizedCrop(
                    image_size,
                    scale=(0.65, 1.0),
                    ratio=(0.75, 1.333),
                ),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(
                    brightness=0.2,
                    contrast=0.2,
                    saturation=0.2,
                    hue=0.05,
                ),
                transforms.ToTensor(),
                normalize,
            ]
        )

    val_transform = transforms.Compose(
        [
            transforms.Resize(resized_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            normalize,
        ]
    )
    return train_transform, val_transform


def _loader_options(
    batch_size: int,
    num_workers: int,
    *,
    seed_workers: bool,
) -> dict:
    options = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": torch.cuda.is_available(),
        "persistent_workers": num_workers > 0,
    }
    if seed_workers:
        options["worker_init_fn"] = _seed_worker
    if num_workers > 0:
        options["prefetch_factor"] = PREFETCH_FACTOR
    return options


def build_data_loaders(
    data_dir: Path,
    batch_size: int,
    image_size: int,
    augment: str,
    val_ratio: float,
    split_seed: int,
    loader_seed: int,
    num_workers: int,
    smoke: bool = False,
):
    train_rows, val_rows, classes = _split_rows(
        Path(data_dir), split_seed, val_ratio, smoke
    )
    train_transform, val_transform = build_transforms(image_size, augment)
    image_dir = Path(data_dir) / "train"
    generator = torch.Generator().manual_seed(loader_seed)
    loader_options = _loader_options(
        batch_size,
        num_workers,
        seed_workers=True,
    )

    train_iter = DataLoader(
        DogDataset(image_dir, train_rows, train_transform),
        shuffle=True,
        drop_last=False,
        generator=generator,
        **loader_options,
    )
    valid_iter = DataLoader(
        DogDataset(image_dir, val_rows, val_transform),
        shuffle=False,
        drop_last=False,
        **loader_options,
    )
    return train_iter, valid_iter, classes


def build_test_loader(
    data_dir: Path,
    batch_size: int,
    image_size: int,
    num_workers: int,
):
    """创建保持文件名排序的测试迭代器，便于生成 Kaggle submission。"""
    _, test_transform = build_transforms(image_size, augment="basic")
    loader_options = _loader_options(
        batch_size,
        num_workers,
        seed_workers=False,
    )
    return DataLoader(
        TestDogDataset(Path(data_dir) / "test", test_transform),
        shuffle=False,
        drop_last=False,
        **loader_options,
    )


def build_full_train_loader(
    data_dir: Path,
    batch_size: int,
    image_size: int,
    augment: str,
    loader_seed: int,
    num_workers: int,
    smoke: bool = False,
):
    """用全部标注图片创建最终训练迭代器，不再保留验证集。"""
    data_dir = Path(data_dir)
    image_ids, targets, classes = _read_labeled_samples(data_dir)
    if smoke:
        image_ids, targets = _limit_smoke_samples(
            image_ids,
            targets,
            loader_seed,
        )
    rows = list(zip(image_ids, targets))
    train_transform, _ = build_transforms(image_size, augment)
    generator = torch.Generator().manual_seed(loader_seed)
    loader_options = _loader_options(
        batch_size,
        num_workers,
        seed_workers=True,
    )
    train_iter = DataLoader(
        DogDataset(data_dir / "train", rows, train_transform),
        shuffle=True,
        drop_last=False,
        generator=generator,
        **loader_options,
    )
    return train_iter, classes


def shutdown_loader(loader) -> None:
    """显式关闭 Windows persistent workers，避免阶段切换时进程累积。"""
    iterator = getattr(loader, "_iterator", None)
    if iterator is not None:
        iterator._shutdown_workers()
        loader._iterator = None
