"""Collect run result files into a ranked CSV/Markdown report."""

from __future__ import annotations

import csv
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
RUNS = HERE / "runs"
SUMMARY_CSV = HERE / "experiment_summary.csv"
EXPERIMENTS_MD = HERE / "EXPERIMENTS.md"
GENERATED_START = "<!-- experiment-summary:start -->"
GENERATED_END = "<!-- experiment-summary:end -->"


def collect_results() -> list[dict]:
    results = []
    for path in RUNS.glob("*/result.json"):
        with path.open(encoding="utf-8") as file:
            item = json.load(file)
        config = item["config"]
        if config.get("smoke", False):
            continue
        results.append(
            {
                "name": item["name"],
                "best_val_acc": item["best_val_acc"],
                "best_val_top5": item["best_val_top5"],
                "best_epoch": item["best_epoch"],
                "epochs_ran": item["epochs_ran"],
                "model": item["model"],
                "augment": item["augment"],
                "lr": config["lr"],
                "backbone_lr": config["backbone_lr"],
                "weight_decay": config["weight_decay"],
                "label_smoothing": config["label_smoothing"],
                "mixup_alpha": config["mixup_alpha"],
                "dropout": config["dropout"],
                "image_size": config["image_size"],
                "batch_size": config["batch_size"],
                "seed": config["seed"],
                "split_seed": config.get("split_seed", config["seed"]),
                "seconds": round(item["seconds"], 1),
            }
        )
    results.sort(key=lambda row: row["best_val_acc"], reverse=True)
    return results


def write_csv(results: list[dict]) -> None:
    with SUMMARY_CSV.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)


def render_markdown(results: list[dict]) -> str:
    lines = [
        GENERATED_START,
        "## 自动生成实验汇总",
        "",
        "按最佳验证准确率排序；`split_seed` 单列展示，避免把不同验证切分误作直接对比。",
        "",
        "| 实验 | Val Acc | Top-5 | 最佳轮 | 模型 | 增强 | 尺寸 | "
        "Head LR | Backbone LR | LS | Mixup | Seed | Split Seed | 耗时(s) |",
        "|---|---:|---:|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in results:
        lines.append(
            f"| {row['name']} | {row['best_val_acc']:.4f} | "
            f"{row['best_val_top5']:.4f} | {row['best_epoch']} | "
            f"{row['model']} | {row['augment']} | {row['image_size']} | "
            f"{row['lr']} | {row['backbone_lr']} | "
            f"{row['label_smoothing']} | {row['mixup_alpha']} | "
            f"{row['seed']} | {row['split_seed']} | {row['seconds']} |"
        )
    lines.append(GENERATED_END)
    return "\n".join(lines)


def update_generated_section(generated: str) -> None:
    current = EXPERIMENTS_MD.read_text(encoding="utf-8")
    if GENERATED_START in current and GENERATED_END in current:
        before, remainder = current.split(GENERATED_START, maxsplit=1)
        _, after = remainder.split(GENERATED_END, maxsplit=1)
        updated = before.rstrip() + "\n\n" + generated + after
    else:
        updated = current.rstrip() + "\n\n" + generated + "\n"
    EXPERIMENTS_MD.write_text(updated, encoding="utf-8")


def main() -> None:
    results = collect_results()
    if not results:
        raise SystemExit("no completed runs found")
    write_csv(results)
    update_generated_section(render_markdown(results))
    print(f"wrote {SUMMARY_CSV}")


if __name__ == "__main__":
    main()
