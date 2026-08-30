"""不重新训练模型的快速行为回归测试。"""

from __future__ import annotations

import csv
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = PROJECT_ROOT.parents[1]
DATA_DIR = REPOSITORY_ROOT / "data" / "dog-breed-identification"
FORMAL_RUN_DIR = PROJECT_ROOT / "runs" / "vit_l16_full5_s2022"
sys.path.insert(0, str(PROJECT_ROOT))

import Data_Prep  # noqa: E402
import main  # noqa: E402
import model  # noqa: E402
import summarize  # noqa: E402


class ProjectRegressionTests(unittest.TestCase):
    def test_cli_defaults_and_choices(self) -> None:
        args = main.parse_args(["--name", "regression"])
        self.assertEqual(args.model, "resnet34")
        self.assertEqual(args.augment, "basic")
        self.assertEqual(args.epochs, 10)
        self.assertEqual(args.batch_size, 256)
        self.assertEqual(args.seed, 42)
        self.assertEqual(
            model.MODEL_NAMES,
            ("legacy_resnet34", "resnet34", "resnet18", "vit_l_16"),
        )

    @unittest.skipUnless(DATA_DIR.is_dir(), "competition data is not available")
    def test_data_split_and_vit_transform_contract(self) -> None:
        data_dir = Data_Prep.load_data(REPOSITORY_ROOT)
        train_rows, valid_rows, classes = Data_Prep._split_rows(
            data_dir,
            split_seed=42,
            val_ratio=0.15,
            smoke=False,
        )
        self.assertEqual(
            (len(train_rows), len(valid_rows), len(classes)),
            (8688, 1534, 120),
        )
        self.assertEqual(
            classes[:3],
            ["affenpinscher", "afghan_hound", "african_hunting_dog"],
        )

        train_transform, valid_transform = Data_Prep.build_transforms(224, "vit")
        self.assertIn("Resize(size=256", repr(train_transform))
        self.assertIn("RandomHorizontalFlip(p=0.6)", repr(train_transform))
        self.assertIn("RandomRotation(degrees=[-30.0, 30.0]", repr(train_transform))
        self.assertNotIn("RandomHorizontalFlip", repr(valid_transform))

    @unittest.skipUnless(
        (FORMAL_RUN_DIR / "submission.csv").is_file(),
        "formal submission artifact is not available",
    )
    def test_formal_submission_and_training_log_contract(self) -> None:
        submission = pd.read_csv(FORMAL_RUN_DIR / "submission.csv")
        template = pd.read_csv(
            REPOSITORY_ROOT
            / "data"
            / "dog-breed-identification"
            / "sample_submission.csv"
        )
        probabilities = submission.iloc[:, 1:].to_numpy()

        self.assertEqual(submission.shape, (10357, 121))
        self.assertTrue(submission["id"].equals(template["id"]))
        self.assertEqual(list(submission.columns), list(template.columns))
        self.assertTrue(np.isfinite(probabilities).all())
        self.assertLess(float(np.abs(probabilities.sum(axis=1) - 1).max()), 1e-5)

        with (PROJECT_ROOT / "TRAINING_LOG.csv").open(encoding="utf-8") as file:
            rows = list(csv.DictReader(file))
        formal_rows = [row for row in rows if row["run"] == "vit_l16_full5_s2022"]
        self.assertEqual(len(formal_rows), 10)
        self.assertEqual(
            [row["phase"] for row in formal_rows],
            ["validation"] * 5 + ["full"] * 5,
        )
        self.assertTrue(
            all(row["train_loss"] and row["train_acc"] for row in formal_rows)
        )

    def test_generated_summary_has_stable_markers(self) -> None:
        results = summarize.collect_results()
        rendered = summarize.render_markdown(results)
        self.assertEqual(rendered.count(summarize.GENERATED_START), 1)
        self.assertEqual(rendered.count(summarize.GENERATED_END), 1)
        self.assertIn("vit_l16_full5_s2022", rendered)


if __name__ == "__main__":
    unittest.main()
