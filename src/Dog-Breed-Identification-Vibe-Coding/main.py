"""Dog Breed 调优副本主入口，沿用原项目的四模块结构。"""

from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path
from typing import Sequence

import numpy as np
import torch

import Data_Prep
from Trainer import Trainer
from model import MODEL_NAMES, build_model, split_parameters


BASE_DIR = Path(__file__).resolve().parents[2]
DEFAULT_RUNS_DIR = Path(__file__).resolve().parent / "runs"
DEFAULT_TRAINING_LOG = Path(__file__).resolve().parent / "TRAINING_LOG.csv"


def set_random_seed(seed: int) -> None:
    """设置当前实验使用的全部随机种子。"""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def RandomSeed(seed: int) -> None:
    """保留原项目公开名称，兼容已有调用。"""
    set_random_seed(seed)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tune pretrained ResNet on dog breeds")
    parser.add_argument("--name", required=True, help="unique run name")
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--runs-dir", type=Path, default=DEFAULT_RUNS_DIR)
    parser.add_argument(
        "--model",
        choices=MODEL_NAMES,
        default="resnet34",
    )
    parser.add_argument(
        "--augment", choices=("original", "basic", "strong", "vit"), default="basic"
    )
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument(
        "--full-epochs",
        type=int,
        default=0,
        help="full-data epochs; 0 uses the best validation epoch",
    )
    parser.add_argument("--freeze-epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--lr", type=float, default=0.003)
    parser.add_argument("--backbone-lr", type=float, default=0.00002)
    parser.add_argument("--weight-decay", type=float, default=0.0001)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--label-smoothing", type=float, default=0.0)
    parser.add_argument("--mixup-alpha", type=float, default=0.2)
    parser.add_argument("--optimizer", choices=("adamw", "sgd"), default="adamw")
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--scheduler", choices=("cosine", "constant"), default="cosine")
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--amp", choices=("off", "fp16", "bf16"), default="bf16")
    parser.add_argument("--early-stopping", type=int, default=0)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument(
        "--skip-predict",
        action="store_true",
        help="skip test inference and submission.csv generation",
    )
    return parser.parse_args(argv)


def validate_args(args: argparse.Namespace) -> None:
    if (
        args.epochs < 1
        or args.full_epochs < 0
        or args.batch_size < 1
        or args.workers < 0
    ):
        raise ValueError("epochs/batch-size must be positive and workers non-negative")
    if not 0 < args.val_ratio < 0.5:
        raise ValueError("val-ratio must be between 0 and 0.5")
    if args.freeze_epochs < 0 or args.freeze_epochs > args.epochs:
        raise ValueError("freeze-epochs must be in [0, epochs]")
    if args.model == "legacy_resnet34":
        args.freeze_epochs = args.epochs
        args.backbone_lr = 0.0


def build_network(args: argparse.Namespace, num_classes: int):
    """保留原入口名称，并将模型选择委托给统一工厂。"""
    return build_model(
        args.model,
        num_classes,
        args.dropout,
    )


def _write_json(path: Path, payload: dict) -> None:
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2)


def _build_config(args: argparse.Namespace, data_dir: Path) -> dict:
    config = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }
    config["data_dir"] = str(data_dir)
    config["training_log_path"] = str(DEFAULT_TRAINING_LOG)
    return config


def _create_trainer(
    args: argparse.Namespace,
    *,
    train_iter,
    valid_iter,
    test_iter,
    classes: list[str],
    run_dir: Path,
    config: dict,
    num_epochs: int,
    freeze_epochs: int,
    early_stopping: int,
) -> Trainer:
    network = build_network(args, len(classes))
    head_parameters, backbone_parameters = split_parameters(network)
    return Trainer(
        net=network,
        train_iter=train_iter,
        valid_iter=valid_iter,
        test_iter=test_iter,
        head_parameters=head_parameters,
        backbone_parameters=backbone_parameters,
        classes=classes,
        run_dir=run_dir,
        config=config,
        lr=args.lr,
        backbone_lr=args.backbone_lr,
        num_epochs=num_epochs,
        freeze_epochs=freeze_epochs,
        weight_decay=args.weight_decay,
        label_smoothing=args.label_smoothing,
        mixup_alpha=args.mixup_alpha,
        optimizer_name=args.optimizer,
        momentum=args.momentum,
        scheduler_name=args.scheduler,
        amp=args.amp,
        early_stopping=early_stopping,
    )


def _shutdown_loaders(*loaders) -> None:
    for loader in loaders:
        if loader is not None:
            Data_Prep.shutdown_loader(loader)


def main(argv: Sequence[str] | None = None) -> dict:
    args = parse_args(argv)
    validate_args(args)
    set_random_seed(args.seed)

    data_dir = args.data_dir or Data_Prep.load_data(BASE_DIR, demo=False)
    run_dir = args.runs_dir / args.name
    run_dir.mkdir(parents=True, exist_ok=False)
    config = _build_config(args, data_dir)
    _write_json(run_dir / "config.json", config)

    train_iter = valid_iter = None
    try:
        train_iter, valid_iter, classes = Data_Prep.build_data_loaders(
            data_dir=data_dir,
            batch_size=args.batch_size,
            image_size=args.image_size,
            augment=args.augment,
            val_ratio=args.val_ratio,
            split_seed=args.split_seed,
            loader_seed=args.seed,
            num_workers=args.workers,
            smoke=args.smoke,
        )
        trainer = _create_trainer(
            args,
            train_iter=train_iter,
            valid_iter=valid_iter,
            test_iter=None,
            classes=classes,
            run_dir=run_dir,
            config=config,
            num_epochs=args.epochs,
            freeze_epochs=args.freeze_epochs,
            early_stopping=args.early_stopping,
        )
        result = trainer.train()
        trainer.plot()
    finally:
        _shutdown_loaders(train_iter, valid_iter)

    full_epochs = args.full_epochs or result["best_epoch"]
    full_freeze_epochs = min(args.freeze_epochs, full_epochs)
    set_random_seed(args.seed)
    full_config = dict(config)
    full_train_iter = None
    try:
        full_train_iter, full_classes = Data_Prep.build_full_train_loader(
            data_dir=data_dir,
            batch_size=args.batch_size,
            image_size=args.image_size,
            augment=args.augment,
            loader_seed=args.seed,
            num_workers=args.workers,
            smoke=args.smoke,
        )
        if full_classes != classes:
            raise ValueError("full training classes do not match validation classes")
        full_config.update(
            {
                "phase": "full",
                "epochs": full_epochs,
                "freeze_epochs": full_freeze_epochs,
                "train_samples": len(full_train_iter.dataset),
            }
        )
        full_trainer = _create_trainer(
            args,
            train_iter=full_train_iter,
            valid_iter=None,
            test_iter=None,
            classes=classes,
            run_dir=run_dir,
            config=full_config,
            num_epochs=full_epochs,
            freeze_epochs=full_freeze_epochs,
            early_stopping=0,
        )
        full_result = full_trainer.train_full(run_dir / "full.pt")
    finally:
        _shutdown_loaders(full_train_iter)

    result["full_training"] = full_result
    if not args.skip_predict:
        test_iter = None
        try:
            test_iter = Data_Prep.build_test_loader(
                data_dir=data_dir,
                batch_size=args.batch_size,
                image_size=args.image_size,
                num_workers=args.workers,
            )
            full_trainer.test_iter = test_iter
            full_trainer.predict(
                run_dir / "submission.csv",
                checkpoint_path=run_dir / "full.pt",
            )
        finally:
            _shutdown_loaders(test_iter)
        result["submission"] = str(run_dir / "submission.csv")
    _write_json(run_dir / "result.json", result)
    print("RESULT " + json.dumps(result, ensure_ascii=False))
    return result


if __name__ == "__main__":
    main()
