"""训练、验证、checkpoint 与曲线记录。"""

from __future__ import annotations

import csv
import json
import math
import platform
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch import nn


TRAINING_LOG_FIELDS = [
    "timestamp",
    "run",
    "model",
    "phase",
    "epoch",
    "lr_head",
    "lr_backbone",
    "train_loss",
    "train_acc",
    "val_loss",
    "val_acc",
    "val_top5",
    "seconds",
    "train_samples",
    "val_samples",
]


def _write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _resolve_devices(devices) -> list[torch.device]:
    if devices is None:
        return [torch.device("cuda:0" if torch.cuda.is_available() else "cpu")]
    if isinstance(devices, (str, torch.device)):
        return [torch.device(devices)]
    return [torch.device(device) for device in devices]


@dataclass
class EpochMetrics:
    epoch: int
    lr_head: float
    lr_backbone: float
    train_loss: float
    train_acc: float
    val_loss: float
    val_acc: float
    val_top5: float
    seconds: float


class Trainer:
    def __init__(
        self,
        net: nn.Module,
        train_iter,
        valid_iter,
        test_iter,
        head_parameters: list[nn.Parameter],
        backbone_parameters: list[nn.Parameter],
        classes: list[str],
        run_dir: Path,
        config: dict,
        lr: float,
        backbone_lr: float,
        num_epochs: int,
        freeze_epochs: int,
        weight_decay: float,
        label_smoothing: float,
        mixup_alpha: float,
        optimizer_name: str = "adamw",
        momentum: float = 0.9,
        scheduler_name: str = "cosine",
        amp: str = "bf16",
        early_stopping: int = 0,
        devices=None,
    ):
        resolved_devices = _resolve_devices(devices)
        if not resolved_devices:
            raise ValueError("at least one device is required")

        self.devices = resolved_devices
        self.device = resolved_devices[0]
        net = net.to(self.device)
        if len(resolved_devices) > 1:
            self.net = nn.DataParallel(
                net,
                device_ids=[device.index for device in resolved_devices],
            )
        else:
            self.net = net

        self.train_iter = train_iter
        self.valid_iter = valid_iter
        self.test_iter = test_iter
        self.head_parameters = head_parameters
        self.backbone_parameters = backbone_parameters
        self.classes = classes
        self.run_dir = Path(run_dir)
        self.config = config
        self.training_log_path = Path(
            config.get(
                "training_log_path",
                self.run_dir.parent.parent / "TRAINING_LOG.csv",
            )
        )
        self.num_epochs = num_epochs
        self.freeze_epochs = freeze_epochs
        self.mixup_alpha = mixup_alpha
        self.early_stopping = early_stopping
        self.history: list[EpochMetrics] = []

        self._set_backbone_trainable(freeze_epochs == 0)
        self.optimizer = self._build_optimizer(
            head_parameters=head_parameters,
            backbone_parameters=backbone_parameters,
            lr=lr,
            backbone_lr=backbone_lr,
            weight_decay=weight_decay,
            optimizer_name=optimizer_name,
            momentum=momentum,
        )
        self.scheduler = self._build_scheduler(scheduler_name)
        self.loss = nn.CrossEntropyLoss(label_smoothing=label_smoothing).to(
            self.device
        )
        self.amp_dtype = None
        if self.device.type == "cuda" and amp != "off":
            self.amp_dtype = torch.bfloat16 if amp == "bf16" else torch.float16
        self.scaler = torch.amp.GradScaler(
            "cuda", enabled=self.amp_dtype == torch.float16
        )

    def _build_optimizer(
        self,
        *,
        head_parameters: list[nn.Parameter],
        backbone_parameters: list[nn.Parameter],
        lr: float,
        backbone_lr: float,
        weight_decay: float,
        optimizer_name: str,
        momentum: float,
    ) -> torch.optim.Optimizer:
        parameter_groups = [
            {"params": backbone_parameters, "lr": backbone_lr},
            {"params": head_parameters, "lr": lr},
        ]
        if optimizer_name == "sgd":
            return torch.optim.SGD(
                parameter_groups,
                momentum=momentum,
                weight_decay=weight_decay,
            )
        return torch.optim.AdamW(
            parameter_groups,
            weight_decay=weight_decay,
        )

    def _build_scheduler(self, scheduler_name: str):
        def cosine_multiplier(epoch: int) -> float:
            return 0.05 + 0.95 * (
                1.0 + math.cos(math.pi * epoch / self.num_epochs)
            ) / 2.0

        multiplier = (
            (lambda _epoch: 1.0)
            if scheduler_name == "constant"
            else cosine_multiplier
        )
        return torch.optim.lr_scheduler.LambdaLR(
            self.optimizer,
            multiplier,
        )

    def _autocast(self):
        return torch.autocast(
            device_type=self.device.type,
            dtype=self.amp_dtype,
            enabled=self.amp_dtype is not None,
        )

    def _set_backbone_trainable(self, trainable: bool) -> None:
        for parameter in self.backbone_parameters:
            parameter.requires_grad = trainable

    def _keep_frozen_batch_norm_eval(self) -> None:
        for module in self.net.modules():
            if isinstance(module, nn.modules.batchnorm._BatchNorm):
                module.eval()

    def _mixup(self, images: torch.Tensor, targets: torch.Tensor):
        if self.mixup_alpha <= 0:
            return images, targets, targets, 1.0
        lam = float(np.random.beta(self.mixup_alpha, self.mixup_alpha))
        indices = torch.randperm(images.size(0), device=images.device)
        mixed = lam * images + (1.0 - lam) * images[indices]
        return mixed, targets, targets[indices], lam

    def _save_history(self) -> None:
        rows = [asdict(item) for item in self.history]
        _write_csv(
            self.run_dir / "history.csv",
            rows,
            list(rows[0]),
        )

    def _append_training_log(self, phase: str, metrics: dict) -> None:
        """将每轮指标追加到项目级日志，便于跨实验比较。"""
        row = {
            "timestamp": datetime.now().astimezone().isoformat(timespec="seconds"),
            "run": self.config["name"],
            "model": self.config["model"],
            "phase": phase,
            "epoch": metrics["epoch"],
            "lr_head": metrics["lr_head"],
            "lr_backbone": metrics["lr_backbone"],
            "train_loss": metrics["train_loss"],
            "train_acc": metrics["train_acc"],
            "val_loss": metrics.get("val_loss", ""),
            "val_acc": metrics.get("val_acc", ""),
            "val_top5": metrics.get("val_top5", ""),
            "seconds": metrics["seconds"],
            "train_samples": len(self.train_iter.dataset),
            "val_samples": (
                len(self.valid_iter.dataset) if self.valid_iter is not None else 0
            ),
        }
        self.training_log_path.parent.mkdir(parents=True, exist_ok=True)
        needs_header = not self.training_log_path.exists()
        with self.training_log_path.open("a", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=TRAINING_LOG_FIELDS)
            if needs_header:
                writer.writeheader()
            writer.writerow(row)

    def evaluate(self):
        self.net.eval()
        total_loss = 0.0
        total_correct = 0
        total_top5 = 0
        total = 0
        with torch.inference_mode():
            for images, targets in self.valid_iter:
                images = images.to(self.device, non_blocking=True)
                targets = targets.to(self.device, non_blocking=True)
                with self._autocast():
                    logits = self.net(images)
                    loss = self.loss(logits, targets)
                batch_size = targets.size(0)
                total_loss += loss.item() * batch_size
                total_correct += (logits.argmax(1) == targets).sum().item()
                total_top5 += (
                    logits.topk(5, dim=1)
                    .indices.eq(targets[:, None])
                    .any(1)
                    .sum()
                    .item()
                )
                total += batch_size
        if total == 0:
            raise ValueError("validation iterator produced no samples")
        return total_loss / total, total_correct / total, total_top5 / total

    def _train_batches(self, train_iter, epoch_index: int):
        if epoch_index == self.freeze_epochs:
            self._set_backbone_trainable(True)
        backbone_frozen = epoch_index < self.freeze_epochs
        self.net.train()
        if backbone_frozen:
            self._keep_frozen_batch_norm_eval()

        total_loss = 0.0
        total_correct = 0.0
        total = 0
        started = time.time()
        for images, targets in train_iter:
            images = images.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)
            images, target_a, target_b, lam = self._mixup(images, targets)
            self.optimizer.zero_grad(set_to_none=True)
            with self._autocast():
                logits = self.net(images)
                loss = (
                    lam * self.loss(logits, target_a)
                    + (1.0 - lam) * self.loss(logits, target_b)
                )
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            batch_size = targets.size(0)
            predictions = logits.argmax(1)
            total_loss += loss.item() * batch_size
            total_correct += (
                lam * (predictions == target_a).sum().item()
                + (1.0 - lam) * (predictions == target_b).sum().item()
            )
            total += batch_size

        if total == 0:
            raise ValueError("training iterator produced no samples")
        return total_loss / total, total_correct / total, time.time() - started

    def train_epoch(self, epoch_index: int) -> EpochMetrics:
        train_loss, train_acc, seconds = self._train_batches(
            self.train_iter, epoch_index
        )
        val_loss, val_acc, val_top5 = self.evaluate()
        return EpochMetrics(
            epoch=epoch_index + 1,
            lr_head=self.optimizer.param_groups[1]["lr"],
            lr_backbone=self.optimizer.param_groups[0]["lr"],
            train_loss=train_loss,
            train_acc=train_acc,
            val_loss=val_loss,
            val_acc=val_acc,
            val_top5=val_top5,
            seconds=seconds,
        )

    def train(self) -> dict:
        best_loss = float("inf")
        best_epoch = 0
        stale_epochs = 0
        started = time.time()
        print(
            f"run={self.config['name']} model={self.config['model']} "
            f"device={self.device} train={len(self.train_iter.dataset)} "
            f"val={len(self.valid_iter.dataset)} classes={len(self.classes)}"
        )

        for epoch_index in range(self.num_epochs):
            metrics = self.train_epoch(epoch_index)
            self.history.append(metrics)
            self._save_history()
            self._append_training_log("validation", asdict(metrics))
            print(
                f"epoch {metrics.epoch:02d}/{self.num_epochs} "
                f"train_loss={metrics.train_loss:.4f} "
                f"train_acc={metrics.train_acc:.4f} "
                f"val_loss={metrics.val_loss:.4f} "
                f"val_acc={metrics.val_acc:.4f} "
                f"top5={metrics.val_top5:.4f} time={metrics.seconds:.1f}s"
            )

            if metrics.val_loss < best_loss:
                best_loss = metrics.val_loss
                best_epoch = metrics.epoch
                stale_epochs = 0
                torch.save(
                    {
                        "model": self.net.state_dict(),
                        "classes": self.classes,
                        "args": self.config,
                        "epoch": metrics.epoch,
                        "val_acc": metrics.val_acc,
                        "val_loss": metrics.val_loss,
                    },
                    self.run_dir / "best.pt",
                )
            else:
                stale_epochs += 1
            self.scheduler.step()
            if self.early_stopping and stale_epochs >= self.early_stopping:
                print(f"early stopping after {stale_epochs} unimproved epochs")
                break

        result = {
            "name": self.config["name"],
            "model": self.config["model"],
            "augment": self.config["augment"],
            "best_epoch": best_epoch,
            "best_val_acc": max(item.val_acc for item in self.history),
            "best_val_loss": best_loss,
            "best_val_top5": max(item.val_top5 for item in self.history),
            "epochs_ran": len(self.history),
            "train_samples": len(self.train_iter.dataset),
            "val_samples": len(self.valid_iter.dataset),
            "trainable_parameters": sum(
                parameter.numel()
                for parameter in self.net.parameters()
                if parameter.requires_grad
            ),
            "total_parameters": sum(
                parameter.numel() for parameter in self.net.parameters()
            ),
            "seconds": time.time() - started,
            "torch": torch.__version__,
            "device": (
                torch.cuda.get_device_name(self.device)
                if self.device.type == "cuda"
                else platform.processor()
            ),
            "config": self.config,
        }
        self._write_json(self.run_dir / "result.json", result)
        return result

    def train_full(self, checkpoint_path: Path | str | None = None) -> dict:
        """不保留验证集，用当前全量训练迭代器训练并保存最终模型。"""
        started = time.time()
        records = []
        print(
            f"run={self.config['name']} phase=full model={self.config['model']} "
            f"device={self.device} train={len(self.train_iter.dataset)} "
            f"classes={len(self.classes)}"
        )
        for epoch_index in range(self.num_epochs):
            train_loss, train_acc, seconds = self._train_batches(
                self.train_iter, epoch_index
            )
            record = {
                "epoch": epoch_index + 1,
                "lr_head": self.optimizer.param_groups[1]["lr"],
                "lr_backbone": self.optimizer.param_groups[0]["lr"],
                "train_loss": train_loss,
                "train_acc": train_acc,
                "seconds": seconds,
            }
            records.append(record)
            self._append_training_log("full", record)
            _write_csv(
                self.run_dir / "full_history.csv",
                records,
                list(record),
            )
            print(
                f"full epoch {epoch_index + 1:02d}/{self.num_epochs} "
                f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
                f"time={seconds:.1f}s"
            )
            self.scheduler.step()

        checkpoint_path = Path(checkpoint_path or self.run_dir / "full.pt")
        torch.save(
            {
                "model": self.net.state_dict(),
                "classes": self.classes,
                "args": self.config,
                "epoch": self.num_epochs,
                "phase": "full",
                "train_samples": len(self.train_iter.dataset),
            },
            checkpoint_path,
        )
        return {
            "epochs": self.num_epochs,
            "train_samples": len(self.train_iter.dataset),
            "final_train_loss": records[-1]["train_loss"],
            "final_train_acc": records[-1]["train_acc"],
            "seconds": time.time() - started,
            "checkpoint": str(checkpoint_path),
        }

    def predict(
        self,
        output_path: Path | str | None = None,
        checkpoint_path: Path | str | None = None,
    ) -> pd.DataFrame:
        """加载本轮最佳权重，对测试集预测并生成 Kaggle 提交文件。"""
        if self.test_iter is None:
            raise ValueError("test iterator is required for prediction")
        checkpoint_path = Path(checkpoint_path or self.run_dir / "best.pt")
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"checkpoint does not exist: {checkpoint_path}")

        checkpoint = torch.load(
            checkpoint_path,
            map_location=self.device,
            weights_only=False,
        )
        checkpoint_classes = checkpoint.get("classes")
        if checkpoint_classes != self.classes:
            raise ValueError(
                "checkpoint classes do not match the current dataset classes"
            )
        self.net.load_state_dict(checkpoint["model"])
        self.net.eval()

        probabilities = []
        test_ids: list[str] = []
        with torch.inference_mode():
            for images, image_ids in self.test_iter:
                images = images.to(self.device, non_blocking=True)
                with self._autocast():
                    logits = self.net(images)
                probabilities.append(torch.softmax(logits.float(), dim=1).cpu())
                test_ids.extend(image_ids)

        if not probabilities:
            raise ValueError("test iterator produced no samples")
        probability_array = torch.cat(probabilities).numpy()
        if probability_array.shape != (len(test_ids), len(self.classes)):
            raise ValueError(
                "prediction shape does not match test IDs and class count: "
                f"{probability_array.shape}"
            )
        if len(test_ids) != len(set(test_ids)):
            raise ValueError("test image IDs are not unique")

        submission = pd.DataFrame(probability_array, columns=self.classes)
        submission.insert(0, "id", test_ids)
        output_path = Path(output_path or self.run_dir / "submission.csv")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        submission.to_csv(output_path, index=False)
        print(
            f"submission={output_path} rows={len(submission)} "
            f"classes={len(self.classes)} checkpoint_epoch={checkpoint['epoch']}"
        )
        return submission

    @staticmethod
    def _write_json(path: Path, payload: dict) -> None:
        with path.open("w", encoding="utf-8") as file:
            json.dump(payload, file, indent=2)

    def plot(self) -> Path:
        if not self.history:
            raise ValueError("there is no training history to plot")
        epochs = [item.epoch for item in self.history]
        figure, axes = plt.subplots(1, 2, figsize=(12, 4))
        axes[0].plot(epochs, [item.train_loss for item in self.history], label="Train")
        axes[0].plot(epochs, [item.val_loss for item in self.history], label="Valid")
        axes[0].set_title("Loss")
        axes[1].plot(epochs, [item.train_acc for item in self.history], label="Train")
        axes[1].plot(epochs, [item.val_acc for item in self.history], label="Valid")
        axes[1].set_title("Accuracy")
        for axis in axes:
            axis.set_xlabel("Epoch")
            axis.grid(True)
            axis.legend()
        figure.tight_layout()
        path = self.run_dir / "training_curves.png"
        figure.savefig(path, dpi=150)
        plt.close(figure)
        return path
