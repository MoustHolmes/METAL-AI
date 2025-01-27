import torchmetrics
from lightning import LightningModule, Trainer
from lightning.pytorch.callbacks import Callback
from torchmetrics.wrappers import MetricTracker
from torchmetrics.classification import BinaryAUROC, BinaryAccuracy, BinaryPrecision, BinaryRecall, BinaryF1Score, BinaryConfusionMatrix
from torchmetrics.regression import MeanSquaredError, MeanAbsoluteError, MeanAbsolutePercentageError, KendallRankCorrCoef


class CSFMetricsLogger(Callback):
    """Callback for logging classification metrics using torchmetrics.
    
    This callback logs metrics at the end of each training, validation, and test batch.
    The metrics are logged to the PyTorch Lightning module's logger.
    
    Args:
        metrics (torchmetrics.MetricCollection): A collection of metrics to log."""
    

    def __init__(self,):
        classification_metrics = torchmetrics.MetricCollection([ BinaryAccuracy(), BinaryPrecision(), BinaryRecall(), BinaryF1Score()])
        self.train_metrics = classification_metrics.clone(prefix="train/")
        self.val_metrics = classification_metrics.clone(prefix="val/")
        self.test_metrics = classification_metrics.clone(prefix="test/")
        prev_batch_size = 0

    # def log_metrics(self, pl_module, metrics, preds, targets):
    #     """Log the given metrics to the PyTorch Lightning module's logger.

    #     Args:
    #         pl_module (LightningModule): The Lightning module being trained.
    #         metrics (torchmetrics.MetricCollection): The metrics to log.
    #         preds (torch.Tensor): The predicted outputs.
    #         targets (torch.Tensor): The ground truth targets.
    #     """

    #     pl_module.log_dict(
    #         metrics(preds, targets),
    #         on_step=False,
    #         on_epoch=True,
    #         prog_bar=True,
    #     )

    def on_train_start(self, trainer, pl_module):
        self.val_metrics.reset()

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        self.train_metrics.update( outputs["preds"], outputs["targets"])
        if outputs["batch_shape"][1] != self.prev_batch_size:
            pl_module.log_dict(self.train_metrics.compute())
            self.train_metrics.reset()

        self.prev_batch_size = outputs["batch_shape"][1]
            
        # self.log_metrics(pl_module, self.train_metrics, outputs["preds"], outputs["targets"])

    def on_validation_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0
    ):
        self.val_metrics.update( outputs["preds"], outputs["targets"])
        if outputs["batch_shape"][1] != self.prev_batch_size:
            pl_module.log_dict(self.val_metrics.compute())
            self.val_metrics.reset()
        self.prev_batch_size = outputs["batch_shape"][1]

    def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        self.test_metrics.update( outputs["preds"], outputs["targets"])
        if outputs["batch_shape"][1] != self.prev_batch_size:
            pl_module.log_dict(self.test_metrics.compute())
            self.test_metrics.reset()
        self.prev_batch_size = outputs["batch_shape"][1]

class RegressionMetricsLogger(Callback):
    """Callback for logging training, validation, and test metrics using torchmetrics.

    This callback logs metrics at the end of each training, validation, and test batch.
    The metrics are logged to the PyTorch Lightning module's logger.

    """

    def __init__(self):# metrics: torchmetrics.MetricCollection

        regression_metrics = torchmetrics.MetricCollection([MeanAbsoluteError(),KendallRankCorrCoef()])# MeanAbsolutePercentageError()
        # self.train_metrics = regression_metrics.clone(prefix="train/")
        # self.val_metrics = regression_metrics.clone(prefix="val/")
        self.test_metrics = regression_metrics.clone(prefix="test/")

    def log_metrics(self, pl_module, metrics, preds, targets):
        """Log the given metrics to the PyTorch Lightning module's logger.

        Args:
            pl_module (LightningModule): The Lightning module being trained.
            metrics (torchmetrics.MetricCollection): The metrics to log.
            preds (torch.Tensor): The predicted outputs.
            targets (torch.Tensor): The ground truth targets.
        """
        # print(preds.shape, targets.shape)
        pl_module.log_dict(
            metrics(preds.view(-1,), targets.view(-1,)),
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )

    def setup(self, trainer, pl_module, stage):
        # self.train_metrics.to(pl_module.device)
        # self.val_metrics.to(pl_module.device)
        self.test_metrics.to(pl_module.device)

    # def on_train_start(self, trainer, pl_module):
    #     self.val_metrics.reset()

    # def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        
    #     self.log_metrics(pl_module, self.train_metrics, outputs["preds"], outputs["targets"])

    # def on_validation_batch_end(
    #     self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0
    # ):
    #     self.log_metrics(pl_module, self.val_metrics, outputs["preds"], outputs["targets"])

    def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        self.log_metrics(pl_module, self.test_metrics, outputs["preds"], outputs["targets"])

class GaussianNLLMetricsLogger(Callback):
    """Callback for logging training, validation, and test metrics using torchmetrics.

    This callback logs metrics at the end of each training, validation, and test batch.
    The metrics are logged to the PyTorch Lightning module's logger.

    """

    def __init__(self):# metrics: torchmetrics.MetricCollection

        regression_metrics = torchmetrics.MetricCollection([MeanAbsoluteError(),])# MeanAbsolutePercentageError()
        # variance_metrics = torchmetrics.MetricCollection([MeanAbsoluteError(),])# MeanAbsolutePercentageError()
        self.train_metrics = regression_metrics.clone(prefix="train/")
        self.train_variance = torchmetrics.MeanMetric()
        self.val_metrics = regression_metrics.clone(prefix="val/")
        self.val_variance = torchmetrics.MeanMetric()
        self.test_metrics = regression_metrics.clone(prefix="test/")
        self.test_variance = torchmetrics.MeanMetric()

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

    def setup(self, trainer, pl_module, stage):
        self.train_metrics.to(pl_module.device)
        self.train_variance.to(pl_module.device)
        self.val_metrics.to(pl_module.device)
        self.val_variance.to(pl_module.device)
        self.test_metrics.to(pl_module.device)
        self.test_variance.to(pl_module.device)

    def on_train_start(self, trainer, pl_module):
        self.val_metrics.reset()
        self.val_variance.reset()

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        mask = outputs["mask"]
        mean = outputs["preds"][:,:,0][mask]
        variance = outputs["preds"][:,:,1][mask]
        targets = outputs["targets"][mask]


        pl_module.log("train/mean_variace", self.test_variance(variance), on_step=False, on_epoch=True, prog_bar=True)
        self.log_metrics(pl_module, self.train_metrics, mean, targets)

    def on_validation_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0
    ):  
        mask = outputs["mask"]
        mean = outputs["preds"][:,:,0][mask]
        variance = outputs["preds"][:,:,1][mask]
        targets = outputs["targets"][mask]

        pl_module.log("val/mean_variace", self.val_variance(variance), on_step=False, on_epoch=True, prog_bar=True)
        self.log_metrics(pl_module, self.val_metrics, mean, targets)

    def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        mask = outputs["mask"]
        mean = outputs["preds"][:,:,0][mask]
        variance = outputs["preds"][:,:,1][mask]
        targets = outputs["targets"][mask]

        pl_module.log("test/mean_variace", self.test_variance(variance), on_step=False, on_epoch=True, prog_bar=True)
        self.log_metrics(pl_module, self.test_metrics, mean, targets)

class ClassificationMetricsLogger(Callback):
    """Callback for logging training, validation, and test metrics using torchmetrics.

    This callback logs metrics at the end of each training, validation, and test batch.
    The metrics are logged to the PyTorch Lightning module's logger.

    """

    def __init__(self):# metrics: torchmetrics.MetricCollection

        # classification_metrics = torchmetrics.MetricCollection([binaryAUROC(),BinaryAccuracy(), BinaryPrecision(), BinaryRecall(), BinaryF1Score()])
        # self.train_metrics = classification_metrics.clone(prefix="train/")
        # self.val_metrics = classification_metrics.clone(prefix="val/")
        self.test_metrics = torchmetrics.MetricCollection([
            BinaryAUROC(), 
            # BinaryConfusionMatrix(), 
            BinaryAccuracy(), 
            BinaryPrecision(), 
            BinaryRecall(), 
            BinaryF1Score()
            ], prefix="test/")

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

    def setup(self, trainer, pl_module, stage):
        # self.train_metrics.to(pl_module.device)
        # self.val_metrics.to(pl_module.device)
        self.test_metrics.to(pl_module.device)

    # def on_train_start(self, trainer, pl_module):
    #     self.val_metrics.reset()

    # def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
    #     self.log_metrics(pl_module, self.train_metrics, outputs["preds"][outputs["mask"]], outputs["targets"][outputs["mask"]])

    # def on_validation_batch_end(
    #     self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0
    # ):
    #     self.log_metrics(pl_module, self.val_metrics, outputs["preds"][outputs["mask"]], outputs["targets"][outputs["mask"]])

    def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        self.log_metrics(pl_module, self.test_metrics, outputs["preds"][outputs["mask"]], outputs["targets"][outputs["mask"]])


class BestMetricsLogger(ClassificationMetricsLogger):
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
