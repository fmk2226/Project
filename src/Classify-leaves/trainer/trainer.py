import numpy as np
from utils import inf_loop,MetricTracker

class Trainer:
    """ Trainer Class

    Args:
        model: PyTorch model to train.
        criterion: Loss function.
        metric_ftns: Named metric functions, such as accuracy & top k accuracy.
        optimizer: Initialized optimizer for the model.
        config: config.json.
        device: CUDA device for training.
        data_loader: Training DataLoader.
        valid_data_loader: Optional validation DataLoader.
        lr_scheduler: Optional learning-rate scheduler.
        len_epoch: Batches per epoch. Defaults to len(data_loader); when specified, the loader is wrapped in an endless iterator.

    Attributes:
        epochs: Total training epochs.
        start_epoch: Starting epoch, initialized to 1.
        do_validation: Whether validation data is available.
        log_step: Logging interval in batches.
        train_metrics: Tracker for training loss and metrics.
        valid_metrics: Tracker for validation loss and metrics.
    """
    def __init__(self,model,criterion,metric_ftns,optimizer,config,device,data_loader,valid_data_loader=None,lr_scheduler=None,len_epoch=None):
        self.model=model
        self.criterion=criterion
        self.metric_ftns=metric_ftns
        self.optimizer=optimizer
        self.config=config
        self.epochs=config["trainer"]["epochs"]
        self.start_epoch=1
        self.device=device
        self.data_loader=data_loader
        if len_epoch is None:
            self.len_epoch=len(self.data_loader)
        else:
            self.data_loader=inf_loop(data_loader)
            self.len_epoch=len_epoch
        self.valid_data_loader=valid_data_loader
        self.do_validation=valid_data_loader is not None
        self.lr_scheduler=lr_scheduler
        self.log_step=int(np.sqrt(data_loader.batch_size))

        metric_names=[metric.__name__ for metric in self.metric_ftns]
        self.train_metrics=MetricTracker("loss",*metric_names)
        self.valid_metrics=MetricTracker("loss",*metric_names)

    def _train_epoch(self,epoch):
        pass

    def _valid_epoch(self,epoch):
        pass

    def _progress(self,batch_idx):
        pass

    def _save_checkpoint(self,epoch,save_best=False):
        pass

    def _resume_checkpoint(self,resume_path):
        pass

    def train(self):
        pass
