# TODO: call set_seed(seed) from utils before building data loaders and model
# TODO: label smoothing - criterion=nn.CrossEntropyLoss(label_smoothing=0.1)
# TODO: warmup + cosine scheduler - timm.scheduler.CosineLRScheduler(warmup_t=...),
#   or torch LinearLR + CosineAnnealingLR chained with SequentialLR
