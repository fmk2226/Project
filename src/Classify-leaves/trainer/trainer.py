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
        # TODO: AMP - create self.scaler=torch.amp.GradScaler('cuda') here
        # TODO: callbacks - add callbacks=None arg, self.callbacks=callbacks or [], and
        #   _notify(hook,**kwargs) that calls getattr(cb,hook)(self,**kwargs) on each callback

    def _train_epoch(self,epoch):
        self.model.train()
        self.train_metrics.reset()
        for batch_idx,(X,y) in enumerate(self.data_loader):
            X,y=X.to(self.device),y.to(self.device)
            self.optimizer.zero_grad()
            # TODO: AMP - wrap forward & loss in torch.autocast('cuda',dtype=torch.float16),
            #   then use self.scaler.scale(loss).backward(), self.scaler.step(self.optimizer), self.scaler.update()
            #   (use torch.amp, torch.cuda.amp is deprecated)
            y_hat=self.model(X)
            loss=self.criterion(y_hat,y)
            loss.backward()
            self.optimizer.step()
            n=X.shape[0]
            self.train_metrics.update('loss',loss.item(),n)
            for met in self.metric_ftns:
                self.train_metrics.update(met.__name__,met(y_hat,y),n)
            if batch_idx % self.log_step==0:
                print(f"Train Epoch: {epoch} [{batch_idx+1}/{self.len_epoch}] Loss: {loss.item():.4f}")
                # TODO: callbacks - replace this print with a PrintCallback, and call
                #   self._notify('on_batch_end',epoch=epoch,batch_idx=batch_idx,loss=loss.item()) every batch
                #   (each callback decides its own logging frequency)
            if batch_idx+1==self.len_epoch:
                break
        log=self.train_metrics.result()
        if self.do_validation:
            val_log=self._valid_epoch(epoch)
            log.update(**{'val_'+k:v for k,v in val_log.items()})
        # TODO: warmup + per-batch cosine - move scheduler step into the batch loop
        #   (e.g. timm CosineLRScheduler: step_update(num_updates) per batch instead of step() per epoch)
        if self.lr_scheduler is not None:
            self.lr_scheduler.step()
        return log

    def _valid_epoch(self,epoch):
        pass

    def _progress(self,batch_idx):
        pass

    def _save_checkpoint(self,epoch,save_best=False):
        # TODO: only store plain Python types (tensors, dict, list, int, float, str) plus label_to_id,
        #   torch.load uses weights_only=True by default since torch 2.6, custom class objects fail to load
        pass

    def _resume_checkpoint(self,resume_path):
        pass

    def train(self):
        # TODO: callbacks - call self._notify('on_epoch_end',epoch=epoch,log=log) after each epoch
        #   and self._notify('on_train_end') after the loop; TensorBoard scalars & parameter
        #   histograms (hist_every) go in TensorBoardCallback (logger/visualization.py), not here
        pass
