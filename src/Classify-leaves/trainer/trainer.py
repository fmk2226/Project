import numpy as np
import torch
from utils import inf_loop,MetricTracker
from pathlib import Path
import time

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
        label_to_id: Mapping from class name to integer IDs
        do_validation: Whether validation data is available.
        log_step: Logging interval in batches.
        train_metrics: Tracker for training loss and metrics.
        valid_metrics: Tracker for validation loss and metrics.
        best_acc: Best validation accuracy
        checkpoint_dir: Folder for saving checkpoints
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
        self.label_to_id=data_loader.label_to_id
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

        self.best_acc=0.0
        self.checkpoint_dir=Path(config['trainer']['save_dir'])
        self.checkpoint_dir.mkdir(parents=True,exist_ok=True)
        # TODO: AMP - create self.scaler=torch.amp.GradScaler('cuda') here
        # TODO: callbacks - add callbacks=None arg, self.callbacks=callbacks or [], and
        #   _notify(hook,**kwargs) that calls getattr(cb,hook)(self,**kwargs) on each callback

    def _train_epoch(self,epoch):
        """ Train the model for one epoch, then validate if a validation loader is given.
        
        Args:
            epoch: Current epoch number, starting from 1.

        Returns:
            A dict of average training loss and metrics.
            if validation is available, validation loss and metrics are also reported.
        """
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
        """ Evaluate the model on validation set without updating weights.
        
        Args:
            epoch: Current epoch number, starting from 1

        Returns:
            A dict of average validation loss and metrics.
        """
        self.model.eval()
        self.valid_metrics.reset()
        with torch.no_grad():
            for batch_idx,(X,y) in enumerate(self.valid_data_loader):
                X,y=X.to(self.device),y.to(self.device)
                y_hat=self.model(X)
                loss=self.criterion(y_hat,y)
                n=X.shape[0]
                self.valid_metrics.update('loss',loss.item(),n)
                for met in self.metric_ftns:
                    self.valid_metrics.update(met.__name__,met(y_hat,y),n)
        return self.valid_metrics.result()

    def _progress(self,batch_idx):
        pass

    def _save_checkpoint(self,epoch,save_best=False):
        """ Save the training state to checkpoint-last.pth, and to model_best.pth if save_best is True.
        
        Args:
            epoch: Current epoch number.
            save_best: Whether this epoch has the best validation accuracy so far.
        """
        state={
            'epoch':epoch,
            'state_dict':self.model.state_dict(),
            'optimizer':self.optimizer.state_dict(),
            'lr_scheduler':self.lr_scheduler.state_dict() if self.lr_scheduler is not None else None,
            'best_acc':self.best_acc,
            'label_to_id':self.label_to_id,
            'config':self.config
        }
        torch.save(state,self.checkpoint_dir/'checkpoint-last.pth')
        if save_best:
            torch.save(state,self.checkpoint_dir/'model_best.pth')

    def _resume_checkpoint(self,resume_path):
        pass

    def train(self):
        """ Train on all epochs, print the log, and record the checkpoint by best validation accuracy """
        # TODO: callbacks - call self._notify('on_epoch_end',epoch=epoch,log=log) after each epoch
        #   and self._notify('on_train_end') after the loop; TensorBoard scalars & parameter
        #   histograms (hist_every) go in TensorBoardCallback (logger/visualization.py), not here
        for epoch in range(self.start_epoch,self.epochs+1):
            start=time.time()
            log=self._train_epoch(epoch)
            print(f"Epoch {epoch}/{self.epochs} ({time.time()-start:.0f}s)")
            for key,value in log.items():
                print(f"    {key:15s}: {value:.4f}")
            is_best=self.do_validation and log['val_accuracy']>self.best_acc
            if is_best:
                self.best_acc=log['val_accuracy']
                print(f"    new best val_accuracy: {self.best_acc:.4f}")
            self._save_checkpoint(epoch,save_best=is_best)