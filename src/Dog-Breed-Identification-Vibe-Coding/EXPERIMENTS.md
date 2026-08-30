# Dog Breed 调参结果

## ViT-L/16：0.1x 路线

参考 Kaggle Code 中 207 票的 `nb.pretrained.vit - Loss : 0.09921`，在现有
PyTorch 四模块框架中加入 `vit_l_16`。只使用比赛训练集，冻结 ImageNet
预训练主干，仅训练 120 类线性头；为适配 RTX 5090，将原方案的
`batch=8, lr=0.009` 线性缩放为 `batch=64, lr=0.072`。

| 实验 | Val Loss | Val Acc | Top-5 | 最佳轮 | 全量轮数 | 全量最终 Loss | 全量最终 Acc |
|---|---:|---:|---:|---:|---:|---:|---:|
| vit_l16_full5_s2022 | **0.10657** | **0.97392** | 0.99870 | 5 | 5 | 0.04126 | 0.98709 |

完整逐轮指标统一追加到项目目录的 `TRAINING_LOG.csv`。正式提交候选为
`runs/vit_l16_full5_s2022/submission.csv`，已通过行数、ID、列顺序、有限值
及逐行概率和校验。Kaggle 提交状态为 Success，Public Log Loss 为
**0.09746**，超过原定 `0.1x` 目标并进入 `0.0x`。

复现命令：

```powershell
$python = 'C:\Users\fmk\.conda\envs\pytorch\python.exe'
& $python .\main.py `
  --name vit_l16_full5_s2022 --model vit_l_16 --augment vit `
  --epochs 5 --full-epochs 5 --freeze-epochs 5 `
  --batch-size 64 --image-size 224 --workers 8 `
  --lr 0.072 --backbone-lr 0 --weight-decay 0 --dropout 0 `
  --label-smoothing 0 --mixup-alpha 0 --optimizer sgd --scheduler constant `
  --seed 2022 --split-seed 2022 --amp bf16
```

<!-- experiment-summary:start -->
## 自动生成实验汇总

按最佳验证准确率排序；`split_seed` 单列展示，避免把不同验证切分误作直接对比。

| 实验 | Val Acc | Top-5 | 最佳轮 | 模型 | 增强 | 尺寸 | Head LR | Backbone LR | LS | Mixup | Seed | Split Seed | 耗时(s) |
|---|---:|---:|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| vit_frozen_5_new | 0.9739 | 0.9987 | 5 | vit_l_16 | vit | 224 | 0.072 | 0.0 | 0.0 | 0.0 | 2022 | 2022 | 90.4 |
| vit_l16_full5_s2022 | 0.9739 | 0.9987 | 5 | vit_l_16 | vit | 224 | 0.072 | 0.0 | 0.0 | 0.0 | 2022 | 2022 | 101.8 |
| vit_l16_lr072_b64_s2022 | 0.9739 | 0.9987 | 5 | vit_l_16 | vit | 224 | 0.072 | 0.0 | 0.0 | 0.0 | 2022 | 2022 | 126.7 |
| vit_unfreeze4_b8 | 0.9648 | 0.9961 | 5 | vit_l_16 | vit | 224 | 0.072 | 2e-05 | 0.0 | 0.0 | 2022 | 2022 | 203.0 |
| final_full_10ep_seed123 | 0.8690 | 0.9850 | 10 | resnet34 | basic | 224 | 0.003 | 2e-05 | 0.0 | 0.2 | 123 | 42 | 121.7 |
| final_mixup02_10ep_seed123 | 0.8690 | 0.9857 | 10 | resnet34 | basic | 224 | 0.003 | 2e-05 | 0.0 | 0.2 | 123 | 42 | 115.7 |
| staged_basic_mixup02_10ep | 0.8683 | 0.9863 | 10 | resnet34 | basic | 224 | 0.003 | 2e-05 | 0.0 | 0.2 | 42 | 42 | 116.7 |
| final_full_10ep_seed42 | 0.8677 | 0.9863 | 10 | resnet34 | basic | 224 | 0.003 | 2e-05 | 0.0 | 0.2 | 42 | 42 | 119.1 |
| staged_basic_10ep | 0.8657 | 0.9863 | 10 | resnet34 | basic | 224 | 0.003 | 2e-05 | 0.0 | 0.0 | 42 | 42 | 114.9 |
| staged_strong_10ep | 0.8657 | 0.9857 | 9 | resnet34 | strong | 224 | 0.003 | 2e-05 | 0.0 | 0.0 | 42 | 42 | 94.6 |
| highres288_mixup02_10ep | 0.8651 | 0.9857 | 10 | resnet34 | basic | 288 | 0.003 | 2e-05 | 0.0 | 0.2 | 42 | 42 | 135.4 |
| direct_head_frozen_5ep | 0.8605 | 0.9883 | 5 | resnet34 | basic | 224 | 0.003 | 3e-05 | 0.0 | 0.0 | 42 | 42 | 73.2 |
| final_mixup02_16ep_seed42 | 0.8585 | 0.9863 | 14 | resnet34 | basic | 224 | 0.003 | 2e-05 | 0.0 | 0.2 | 42 | 42 | 159.7 |
| baseline_legacy_5ep | 0.8416 | 0.9844 | 5 | legacy_resnet34 | original | 224 | 0.0005 | 0.0 | 0.0 | 0.0 | 42 | 42 | 65.7 |
| direct_head_basic_6ep | 0.8325 | 0.9811 | 5 | resnet34 | basic | 224 | 0.0008 | 8e-05 | 0.1 | 0.0 | 42 | 42 | 82.8 |
<!-- experiment-summary:end -->
