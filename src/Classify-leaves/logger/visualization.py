# TODO: TensorBoardCallback(log_dir,hist_every=0) - SummaryWriter scalars per batch/epoch,
#   parameter histograms every hist_every epochs, close writer on train end
import logging

class Callback:
    """Base class for callbacks that observe training events in one fold.
    Subclasses can override the hook methods.
    """
    def on_batch_end(self,trainer,epoch,batch_idx,loss):
        """Run after one batch has updated the model.

        Args:
            trainer: Trainer object.
            epoch: Current epoch number, starting at 1.
            batch_idx: Index of the completed batch within this epoch, starting at 0.
            loss: Loss for current batch
        """
        pass

    def on_epoch_end(self,trainer,epoch,log,elapsed,is_best):
        """Run after one epoch checkpoint has saved.
        
        Args:
            trainer: Trainer object.
            epoch: Current epoch number, starting at 1.
            log: Epoch metrics. Validation keys have a 'val_' prefix.
            elapsed: Time for training & validating current epoch.
            is_best: Boolean indicates whether validation accuracy exceeded the previous best.
        """
        pass

    def on_train_end(self,trainer):
        """Run when training exists, including after an interruption or error.
        
        Args:
            trainer: Trainer object.
        """
        pass

class PrintCallback(Callback):
    """Write training progress and epoch metrics using the "train" logger """
    def __init__(self):
        self.logger=logging.getLogger("train")

    def on_batch_end(self,trainer,epoch,batch_idx,loss):
        """Log loss every trainer.log_step batches and on the final batch. """
        batch=batch_idx+1
        if batch % trainer.log_step==0 or batch==trainer.len_epoch:
            self.logger.info("Epoch %d | batch %d/%d | loss %.4f",epoch,batch,trainer.len_epoch,loss)

    def on_epoch_end(self,trainer,epoch,log,elapsed,is_best):
        """Log epoch duration, metrics, and any new best validation accuracy. """
        metrics=" | ".join(f"{name}={value:.4f}" for name,value in log.items())
        self.logger.info("Epoch %d/%d (%.1fs) | %s",epoch,trainer.epochs,elapsed,metrics)
        if is_best:
            self.logger.info("New best validation accuracy: %.4f",log["val_accuracy"])