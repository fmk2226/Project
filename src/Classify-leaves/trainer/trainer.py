import numpy as np
from utils import inf_loop,MetricTracker

class Trainer:
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