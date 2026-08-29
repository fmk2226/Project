from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import torch
from torch import nn

class Trainer:
    def __init__(self, net, train_iter, train_valid_iter, valid_iter, test_iter,
                 batch_size, lr, lr_period, lr_decay, num_epochs, weight_decay,
                 optimizer=None, loss=None, devices=None):
        if devices is None:
            if torch.cuda.is_available():
                devices = [
                    torch.device(f"cuda:{i}")
                    for i in range(torch.cuda.device_count())
                ]
            else:
                devices = [torch.device("cpu")]
        elif isinstance(devices, (str, torch.device)):
            devices = [torch.device(devices)]
        else:
            devices = [torch.device(device) for device in devices]

        if not devices:
            devices = [torch.device("cpu")]

        self.devices = devices
        self.device = devices[0]
        net = net.to(self.device)
        if len(devices) > 1:
            if any(device.type != "cuda" for device in devices):
                raise ValueError("DataParallel only supports CUDA devices")
            device_ids = [device.index for device in devices]
            self.net = nn.DataParallel(net, device_ids=device_ids)
        else:
            self.net = net

        self.train_iter = train_iter
        self.train_valid_iter = train_valid_iter
        self.valid_iter = valid_iter
        self.test_iter = test_iter
        self.batch_size = batch_size
        self.lr = lr
        self.lr_period = lr_period
        self.lr_decay = lr_decay
        self.num_epochs = num_epochs
        self.weight_decay = weight_decay
        optimizer_cls = torch.optim.AdamW if optimizer is None else optimizer
        trainable_params = [param for param in self.net.parameters()
                            if param.requires_grad]
        if not trainable_params:
            raise ValueError("The model has no trainable parameters")
        self.optimizer = optimizer_cls(trainable_params, lr=self.lr, weight_decay=self.weight_decay)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=self.lr_period, gamma=self.lr_decay)

        if loss is None:
            self.loss = nn.CrossEntropyLoss()
        elif isinstance(loss, nn.Module):
            self.loss = loss
        else:
            self.loss = loss()
        if isinstance(self.loss, nn.Module):
            self.loss = self.loss.to(self.device)

        self.history = {
            "train_loss": [],
            "train_acc": [],
            "train_valid_loss": [],
            "train_valid_acc": [],
            "val_loss": [],
            "val_acc": [],
        }

    def accuracy(self, y_hat, y):
        predictions = y_hat.argmax(dim=1)
        return (predictions.to(y.dtype) == y).sum()

    def train_epoch(self, train_iter, valid_iter=None):
        is_train_valid = train_iter is self.train_valid_iter and valid_iter is None
        history_prefix = "train_valid" if is_train_valid else "train"

        for epoch in range(self.num_epochs):
            self.net.train()
            metric_loss = torch.zeros((),device=self.device)
            metric_acc = torch.zeros((),device=self.device,dtype=torch.long)
            metric_total = 0

            for X, y in train_iter:
                X = X.to(self.device, non_blocking=True)
                y = y.to(self.device, non_blocking=True)

                self.optimizer.zero_grad(set_to_none=True)
                y_hat = self.net(X)
                batch_loss = self.loss(y_hat, y)
                batch_loss.backward()
                self.optimizer.step()

                batch_size = y.shape[0]
                metric_loss += batch_loss.detach() * batch_size
                metric_acc += self.accuracy(y_hat.detach(), y)
                metric_total += batch_size

            if metric_total == 0:
                raise ValueError(
                    "The training iterator produced no samples; check batch_size "
                    "and drop_last"
                )

            train_loss = (metric_loss / metric_total).item()
            train_acc = (metric_acc / metric_total).item()
            self.history[f"{history_prefix}_loss"].append(train_loss)
            self.history[f"{history_prefix}_acc"].append(train_acc)

            message = (
                f"epoch {epoch + 1}, "
                f"train_loss: {train_loss:.4f}, "
                f"train_acc: {train_acc:.4f}"
            )
            if valid_iter is not None:
                val_loss, val_acc = self.evaluate(valid_iter)
                self.history["val_loss"].append(val_loss)
                self.history["val_acc"].append(val_acc)
                message += f", val_loss: {val_loss:.4f}, val_acc: {val_acc:.4f}"
            print(message)
            self.scheduler.step()

    def evaluate(self, valid_iter):
        self.net.eval()
        metric_loss = torch.zeros((),device=self.device)
        metric_acc = torch.zeros((),device=self.device,dtype=torch.long)
        metric_total = 0

        with torch.no_grad():
            for X, y in valid_iter:
                X = X.to(self.device, non_blocking=True)
                y = y.to(self.device, non_blocking=True)
                y_hat = self.net(X)
                batch_loss = self.loss(y_hat, y)

                batch_size = y.shape[0]
                metric_loss += batch_loss * batch_size
                metric_acc += self.accuracy(y_hat, y)
                metric_total += batch_size

        if metric_total == 0:
            raise ValueError("The validation iterator produced no samples")
        return (metric_loss / metric_total).item(), (metric_acc / metric_total).item()

    def train(self):
        self.train_epoch(self.train_iter, self.valid_iter)
        return self.history

    def predict(self, test_ds, train_valid_ds, output_path="submission.csv"):
        self.train_epoch(self.train_valid_iter, None)
        self.net.eval()
        probabilities = []

        with torch.no_grad():
            for X, _ in self.test_iter:
                y_hat = self.net(X.to(self.device, non_blocking=True))
                probabilities.append(torch.softmax(y_hat, dim=1).cpu())

        if not probabilities:
            raise ValueError("The test iterator produced no samples")
        probabilities = torch.cat(probabilities).numpy()
        classes = list(train_valid_ds.classes)
        if probabilities.shape[1] != len(classes):
            raise ValueError(
                f"Model output has {probabilities.shape[1]} classes, but the "
                f"dataset has {len(classes)} classes"
            )

        # ImageFolder and the non-shuffled test DataLoader use this same order.
        test_ids = [Path(path).stem for path, _ in test_ds.samples]
        if len(test_ids) != len(probabilities):
            raise ValueError("Test dataset and prediction counts do not match")

        submission = pd.DataFrame(probabilities, columns=classes)
        submission.insert(0, "id", test_ids)
        submission.to_csv(output_path, index=False)
        return submission

    def plot(self, val=True):
        use_train_valid = not val and bool(self.history["train_valid_loss"])
        prefix = "train_valid" if use_train_valid else "train"
        losses = self.history[f"{prefix}_loss"]
        accuracies = self.history[f"{prefix}_acc"]
        if not losses:
            raise ValueError("There is no training history to plot")

        epochs = range(1, len(losses) + 1)
        train_label = "Train+Valid" if use_train_valid else "Train"

        plt.figure(figsize=(8, 5))
        plt.plot(epochs, accuracies, label=f"{train_label} ACC")
        if val:
            plt.plot(epochs, self.history["val_acc"], label="Val ACC")
        plt.xlabel("Epochs")
        plt.ylabel("ACC")
        plt.legend()
        plt.title("ACC vs Epochs")
        plt.grid(True)
        plt.show()

        plt.figure(figsize=(8, 5))
        plt.plot(epochs, losses, label=f"{train_label} Loss")
        if val:
            plt.plot(epochs, self.history["val_loss"], label="Val Loss")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()
        plt.title("Loss vs Epochs")
        plt.grid(True)
        plt.show()
