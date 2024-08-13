import torchmetrics
from lightning import LightningModule, Trainer
from lightning.pytorch.callbacks import Callback
from torchmetrics.wrappers import MetricTracker


class MetricsLogger(Callback):
    """Callback for logging training, validation, and test metrics using torchmetrics.

    This callback logs metrics at the end of each training, validation, and test batch.
    The metrics are logged to the PyTorch Lightning module's logger.

    Args:
        metrics (torchmetrics.MetricCollection): A collection of metrics to log.

    Attributes:
        train_metrics (torchmetrics.MetricCollection): Metrics for tracking training performance.
        val_metrics (torchmetrics.MetricCollection): Metrics for tracking validation performance.
        test_metrics (torchmetrics.MetricCollection): Metrics for tracking test performance.
    """

    def __init__(self, metrics: torchmetrics.MetricCollection):
        self.train_metrics = metrics.clone(prefix="train/")
        self.val_metrics = metrics.clone(prefix="val/")
        self.test_metrics = metrics.clone(prefix="test/")

    def log_metrics(self, pl_module, metrics, preds, targets):
        """Log the given metrics to the PyTorch Lightning module's logger.

        Args:
            pl_module (LightningModule): The Lightning module being trained.
            metrics (torchmetrics.MetricCollection): The metrics to log.
            preds (torch.Tensor): The predicted outputs.
            targets (torch.Tensor): The ground truth targets.
        """

        pl_module.log_dict(
            metrics(preds, targets),
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )

    def on_train_start(self, trainer, pl_module):
        self.val_metrics.reset()

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        self.log_metrics(pl_module, self.train_metrics, outputs["preds"], outputs["targets"])

    def on_validation_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0
    ):
        self.log_metrics(pl_module, self.val_metrics, outputs["preds"], outputs["targets"])

    def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        self.log_metrics(pl_module, self.test_metrics, outputs["preds"], outputs["targets"])


class BestMetricsLogger(MetricsLogger):
    """Callback for logging and tracking the best validation metrics using torchmetrics
    MetricTracker.

    This callback extends MetricLogger to track the best validation metrics over epochs.
    It logs the best metrics observed so far at the end of each validation epoch.

    Args:
        metrics (torchmetrics.MetricCollection): A collection of metrics to log and track.

    Attributes:
        val_metrics (torchmetrics.MetricTracker): Metrics for tracking the best validation performance.
    """

    def __init__(self, metrics: torchmetrics.MetricCollection):
        super().__init__(metrics)
        self.val_metrics = MetricTracker(
            metrics.clone(prefix="val/"),
            maximize=[metric.higher_is_better for _, metric in metrics.items()],
        )

    def on_validation_epoch_start(self, trainer, pl_module):
        self.val_metrics.increment()

    def on_validation_epoch_end(self, trainer, pl_module):
        pl_module.log_dict(
            {f"{k}_best": v for k, v in self.val_metrics.best_metric().items()},
            prog_bar=True,
        )