# Dog Breed Identification — Vibe Coding

这是 `Dog-Breed-Identification` 的独立实验副本。它共享只读的原始数据目录，但不依赖原项目代码，也不会创建 `train_valid_test` 数据副本。代码沿用原项目的 `main.py / Trainer.py / model.py / Data_Prep.py` 四模块结构；`train.py` 仅作为旧命令的兼容入口。

主要修正：

- 将 ResNet34 的最终 `fc` 直接替换为 120 类分类头，不再把 ImageNet 1000 类 logits 当特征；
- 使用分层随机训练/验证切分，所有实验固定 seed=42；
- 支持先冻结、再用差分学习率解冻主干；
- 支持 basic/strong 增强、label smoothing、Mixup、AdamW、余弦退火和 BF16 AMP；
- 新增与 Kaggle 高票方案一致的冻结 ViT-L/16、`vit` 预处理、SGD 和常量学习率；
- 每轮保存 `config.json`、`history.csv`、`result.json` 和最佳 checkpoint；最佳权重按比赛指标 `val_loss` 选择；
- 所有实验的逐轮 acc/loss 同时追加到项目级 `TRAINING_LOG.csv`，便于跨 run 比较。
- `--split-seed` 与 `--seed` 分离，可固定验证集后单独复验训练随机性。

模块职责与原项目一致：

- `main.py`：参数配置、随机种子和训练流程编排；
- `Data_Prep.py`：标签读取、分层切分、预处理和数据增强；
- `model.py`：ResNet18/34、ViT-L/16、分类头和差分学习率参数分组；
- `Trainer.py`：训练验证、冻结/解冻、Mixup、日志、checkpoint、预测和曲线；
- `train.py`：兼容旧调用方式，转发到 `main.py`。

工程约定：

- 模型名称和构造统一由 `model.MODEL_NAMES / model.build_model()` 管理；
- `main.py` 只负责编排验证、全量训练和预测，并在每个阶段可靠释放 DataLoader workers；
- `summarize.py` 只更新 `EXPERIMENTS.md` 中带标记的自动生成区域，不覆盖人工实验说明；
- `pyproject.toml` 统一 Black/Ruff 的 Python 3.10、88 字符规范；
- `tests/test_regression.py` 提供不重新训练模型的快速行为回归检查。

运行环境：

```powershell
$python = 'C:\Users\fmk\.conda\envs\pytorch\python.exe'
& $python .\src\Dog-Breed-Identification-Vibe-Coding\main.py --name my_run
& $python .\src\Dog-Breed-Identification-Vibe-Coding\summarize.py
& $python -m unittest discover .\src\Dog-Breed-Identification-Vibe-Coding\tests -v
```

`legacy_resnet34` 用于在相同数据切分上复现原项目的“冻结 ImageNet 1000 类输出再接 MLP”设计。所有实验产物位于 `runs/<name>/`，汇总结果见 `EXPERIMENTS.md`。

`main.py` 先用训练/验证切分确定最佳轮数，再重新初始化模型，用全部 10,222 张标注图片训练；最终权重保存为 `runs/<name>/full.pt`。随后默认调用 `trainer.predict()` 加载 `full.pt`，并在 `runs/<name>/submission.csv` 生成 10,357 行、120 个犬种概率列的 Kaggle 提交文件。`--full-epochs 0` 表示自动采用最佳验证轮数，也可传入正整数显式指定；仅需训练而不预测时可传入 `--skip-predict`。

## 当前最佳配置

当前最佳验证结果来自冻结 ImageNet 预训练 ViT-L/16：`val_loss=0.10657`、
`val_acc=97.39%`。完整全量训练 5 轮后 `train_loss=0.04126`、
`train_acc=98.71%`，候选文件为
`runs/vit_l16_full5_s2022/submission.csv`；Kaggle Public Log Loss 为
**0.09746**。详见 `EXPERIMENTS.md`、`KAGGLE_RESULTS.md` 与
`TRAINING_LOG.csv`。
复现命令:
"C:\Users\fmk\.conda\envs\pytorch\python.exe" "H:\deeplearning\Project\src\Dog-Breed-Identification-Vibe-Coding\main.py" --name vit_l16_full5_s2022_reproduce --model vit_l_16 --augment vit --epochs 5 --full-epochs 5 --freeze-epochs 5 --batch-size 64 --image-size 224 --workers 8 --lr 0.072 --backbone-lr 0 --weight-decay 0 --dropout 0 --label-smoothing 0 --mixup-alpha 0 --optimizer sgd --momentum 0.9 --scheduler constant --val-ratio 0.15 --seed 2022 --split-seed 2022 --amp bf16

以下为此前 ResNet34 路线：

两次固定验证集复验分别得到 86.83%（seed=42）和 86.90%（seed=123），均值 86.86%；原设计基线为 84.16%。最佳单次 checkpoint 位于 `runs/final_mixup02_10ep_seed123/best.pt`。

最终正式模型已用全部 10,222 张标注图片训练 10 轮，权重位于 `runs/final_full_10ep_seed123/full.pt`，对应预测文件位于 `runs/final_full_10ep_seed123/submission.csv`。

进一步使用 seed=42/123 两个全量模型、水平翻转 TTA 和验证集温度校准（T=0.63）后，Kaggle Public Log Loss 从 0.54583 降至 0.42594。详见 `KAGGLE_RESULTS.md`，复现脚本为 `optimize_submission.py`。

```powershell
$python = 'C:\Users\fmk\.conda\envs\pytorch\python.exe'
& $python .\src\Dog-Breed-Identification-Vibe-Coding\main.py `
  --name reproduce_best --model resnet34 --augment basic `
  --epochs 10 --freeze-epochs 3 --batch-size 256 --image-size 224 --workers 8 `
  --lr 0.003 --backbone-lr 0.00002 --weight-decay 0.0001 `
  --dropout 0.2 --label-smoothing 0 --mixup-alpha 0.2 `
  --seed 123 --split-seed 42
```
