# Kaggle 提交结果

比赛：Dog Breed Identification

| 提交 | 方法 | Public Log Loss |
|---|---|---:|
| submission.csv | 单模型、全量训练 | 0.54583 |
| submission_tta_ensemble_t063.csv | ResNet34 双种子全量模型、水平翻转 TTA、温度校准 T=0.63 | 0.42594 |
| runs/vit_l16_full5_s2022/submission.csv | 冻结 ImageNet ViT-L/16，全量训练线性头 5 轮 | **0.09746** |

新方案在固定验证集上的校准 log loss 为 0.44050。Kaggle Public Log Loss 相比 0.54583 下降 0.11989，相对改善约 21.96%。

复现入口为 `optimize_submission.py`；校准与模型路径记录在 `runs/ensemble_seed42_seed123/submission_tta_ensemble_t063.json`。

## ViT-L/16 最终结果

- Kaggle 状态：`Success / Complete (after deadline)`
- 提交说明：`ViT-L/16 ImageNet frozen backbone, full-data 5 epochs, SGD lr=0.072`
- 本地固定验证集：log loss `0.10657`，accuracy `0.97392`
- Kaggle Public Log Loss：`0.09746`
- 相比 ResNet34 集成的 `0.42594` 下降 `0.32848`，相对改善约 `77.12%`
- 相比最初的 `0.54583` 相对改善约 `82.15%`

训练与复现细节见 `EXPERIMENTS.md`，逐轮 acc/loss 见 `TRAINING_LOG.csv`。
