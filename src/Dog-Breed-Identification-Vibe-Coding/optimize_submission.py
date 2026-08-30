"""用验证集选择温度，并对多个全量模型进行 TTA 概率集成。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import Data_Prep
from model import build_model as create_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Calibrate and ensemble dog-breed models"
    )
    parser.add_argument("--validation-checkpoints", type=Path, nargs="+", required=True)
    parser.add_argument("--full-checkpoints", type=Path, nargs="*", default=[])
    parser.add_argument("--output", type=Path)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--temperature-min", type=float, default=0.50)
    parser.add_argument("--temperature-max", type=float, default=1.50)
    parser.add_argument("--temperature-steps", type=int, default=101)
    return parser.parse_args()


def load_checkpoint(path: Path) -> dict:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    required_keys = {"model", "classes", "args"}
    if not required_keys.issubset(checkpoint):
        raise ValueError(f"invalid checkpoint: {path}")
    return checkpoint


def build_model(checkpoint, device: torch.device):
    config = checkpoint["args"]
    dropout = float(config.get("dropout", 0.2))
    network = create_model(
        config["model"],
        len(checkpoint["classes"]),
        dropout,
    )
    network.load_state_dict(checkpoint["model"])
    network.to(device).eval()
    return network


def collect_logits(
    checkpoint_path: Path,
    loader,
    device: torch.device,
):
    checkpoint = load_checkpoint(checkpoint_path)
    network = build_model(checkpoint, device)
    center_logits = []
    flipped_logits = []
    labels_or_ids = []
    amp_dtype = torch.bfloat16 if device.type == "cuda" else None
    with torch.inference_mode():
        for images, batch_labels_or_ids in loader:
            images = images.to(device, non_blocking=True)
            with torch.autocast(
                device_type=device.type,
                dtype=amp_dtype,
                enabled=amp_dtype is not None,
            ):
                center = network(images)
                flipped = network(torch.flip(images, dims=[3]))
            center_logits.append(center.float().cpu())
            flipped_logits.append(flipped.float().cpu())
            if torch.is_tensor(batch_labels_or_ids):
                labels_or_ids.extend(batch_labels_or_ids.tolist())
            else:
                labels_or_ids.extend(batch_labels_or_ids)
    del network
    if device.type == "cuda":
        torch.cuda.empty_cache()
    logits = torch.stack(
        [torch.cat(center_logits), torch.cat(flipped_logits)], dim=1
    )
    return logits, labels_or_ids, checkpoint["classes"]


def ensemble_probabilities(
    logits_by_model: list[torch.Tensor],
    temperature: float,
    tta: bool,
) -> torch.Tensor:
    probabilities = []
    for logits in logits_by_model:
        selected = logits if tta else logits[:, :1]
        probabilities.append(torch.softmax(selected / temperature, dim=-1).mean(dim=1))
    return torch.stack(probabilities).mean(dim=0)


def log_loss(probabilities: torch.Tensor, labels: torch.Tensor) -> float:
    selected = probabilities[torch.arange(len(labels)), labels].clamp_min(1e-12)
    return -selected.log().mean().item()


def accuracy(probabilities: torch.Tensor, labels: torch.Tensor) -> float:
    return (probabilities.argmax(dim=1) == labels).float().mean().item()


def find_temperature(
    logits_by_model,
    labels,
    low: float,
    high: float,
    steps: int,
):
    best_temperature = 1.0
    best_loss = float("inf")
    for temperature in np.linspace(low, high, steps):
        probabilities = ensemble_probabilities(
            logits_by_model, float(temperature), tta=True
        )
        loss = log_loss(probabilities, labels)
        if loss < best_loss:
            best_temperature = float(temperature)
            best_loss = loss
    return best_temperature, best_loss


def main() -> dict:
    args = parse_args()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    validation_checkpoints = [
        load_checkpoint(path) for path in args.validation_checkpoints
    ]
    classes = validation_checkpoints[0]["classes"]
    if any(checkpoint["classes"] != classes for checkpoint in validation_checkpoints):
        raise ValueError("validation checkpoint classes do not match")
    config = validation_checkpoints[0]["args"]
    data_dir = Path(config["data_dir"])
    unused_train_iter, valid_iter, valid_classes = Data_Prep.build_data_loaders(
        data_dir=data_dir,
        batch_size=args.batch_size,
        image_size=int(config["image_size"]),
        augment=config["augment"],
        val_ratio=float(config["val_ratio"]),
        split_seed=int(config.get("split_seed", config["seed"])),
        loader_seed=int(config["seed"]),
        num_workers=args.workers,
        smoke=False,
    )
    if valid_classes != classes:
        raise ValueError("validation data classes do not match checkpoints")

    validation_logits = []
    validation_labels = None
    for path in args.validation_checkpoints:
        logits, labels, checkpoint_classes = collect_logits(path, valid_iter, device)
        if checkpoint_classes != classes:
            raise ValueError(f"class mismatch: {path}")
        label_tensor = torch.tensor(labels, dtype=torch.long)
        if validation_labels is None:
            validation_labels = label_tensor
        elif not torch.equal(validation_labels, label_tensor):
            raise ValueError("validation label order changed between model passes")
        validation_logits.append(logits)
    Data_Prep.shutdown_loader(unused_train_iter)
    Data_Prep.shutdown_loader(valid_iter)

    individual = []
    for path, logits in zip(args.validation_checkpoints, validation_logits):
        center = ensemble_probabilities([logits], temperature=1.0, tta=False)
        tta = ensemble_probabilities([logits], temperature=1.0, tta=True)
        item = {
            "checkpoint": str(path),
            "center_log_loss": log_loss(center, validation_labels),
            "tta_log_loss": log_loss(tta, validation_labels),
            "center_accuracy": accuracy(center, validation_labels),
            "tta_accuracy": accuracy(tta, validation_labels),
        }
        individual.append(item)
        print(json.dumps(item))

    ensemble_center = ensemble_probabilities(validation_logits, 1.0, tta=False)
    ensemble_tta = ensemble_probabilities(validation_logits, 1.0, tta=True)
    best_temperature, calibrated_loss = find_temperature(
        validation_logits,
        validation_labels,
        args.temperature_min,
        args.temperature_max,
        args.temperature_steps,
    )
    calibration = {
        "models": len(validation_logits),
        "ensemble_center_log_loss": log_loss(ensemble_center, validation_labels),
        "ensemble_tta_log_loss": log_loss(ensemble_tta, validation_labels),
        "ensemble_tta_accuracy": accuracy(ensemble_tta, validation_labels),
        "temperature": best_temperature,
        "calibrated_log_loss": calibrated_loss,
    }
    print("CALIBRATION " + json.dumps(calibration))

    result = {"individual": individual, "calibration": calibration}
    if args.full_checkpoints:
        if len(args.full_checkpoints) != len(args.validation_checkpoints):
            raise ValueError("full and validation checkpoint counts must match")
        if args.output is None:
            raise ValueError("--output is required with --full-checkpoints")
        test_iter = Data_Prep.build_test_loader(
            data_dir=data_dir,
            batch_size=args.batch_size,
            image_size=int(config["image_size"]),
            num_workers=args.workers,
        )
        test_logits = []
        test_ids = None
        for path in args.full_checkpoints:
            logits, ids, checkpoint_classes = collect_logits(path, test_iter, device)
            if checkpoint_classes != classes:
                raise ValueError(f"class mismatch: {path}")
            if test_ids is None:
                test_ids = ids
            elif test_ids != ids:
                raise ValueError("test ID order changed between model passes")
            test_logits.append(logits)
        Data_Prep.shutdown_loader(test_iter)
        probabilities = ensemble_probabilities(
            test_logits, best_temperature, tta=True
        ).numpy()
        submission = pd.DataFrame(probabilities, columns=classes)
        submission.insert(0, "id", test_ids)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        submission.to_csv(args.output, index=False)
        metadata_path = args.output.with_suffix(".json")
        result.update(
            {
                "full_checkpoints": [
                    str(path) for path in args.full_checkpoints
                ],
                "output": str(args.output),
                "rows": len(submission),
            }
        )
        metadata_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(f"submission={args.output} rows={len(submission)} classes={len(classes)}")
    return result


if __name__ == "__main__":
    main()
