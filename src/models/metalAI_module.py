from typing import Any, Dict, Tuple
import pickle
import copy
import torch
from lightning import LightningModule
from torchmetrics import MaxMetric, MeanMetric
from torchmetrics.classification.accuracy import Accuracy


class MetalAILitModule(LightningModule):
    """Example of a `LightningModule` for MNIST classification.

    A `LightningModule` implements 8 key methods:

    ```python
    def __init__(self):
    # Define initialization code here.

    def setup(self, stage):
    # Things to setup before each stage, 'fit', 'validate', 'test', 'predict'.
    # This hook is called on every process when using DDP.

    def training_step(self, batch, batch_idx):
    # The complete training step.

    def validation_step(self, batch, batch_idx):
    # The complete validation step.

    def test_step(self, batch, batch_idx):
    # The complete test step.

    def predict_step(self, batch, batch_idx):
    # The complete predict step.

    def configure_optimizers(self):
    # Define and configure optimizers and LR schedulers.
    ```

    Docs:
        https://lightning.ai/docs/pytorch/latest/common/lightning_module.html
    """

    def __init__(
        self,
        model: torch.nn.Module,
        loss_fn: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        compile: bool,
    ) -> None:
        """Initialize a `MNISTLitModule`.

        :param net: The model to train.
        :param optimizer: The optimizer to use for training.
        :param scheduler: The learning rate scheduler to use for training.
        """
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        self.model = model

        # loss function
        # self.criterion = torch.nn.CrossEntropyLoss()

        # loss function
        self.loss_fn = loss_fn  # torch.nn.MSELoss()
        # print(self.loss_fn)
        # metric objects for calculating and averaging accuracy across batches
        self.train_acc = Accuracy(task="binary")
        self.val_acc = Accuracy(task="binary")
        self.test_acc = Accuracy(task="binary")

        # for averaging loss across batches
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()

        self.pred_dict = {}

        # for tracking best so far validation accuracy
        # self.val_acc_best = MaxMetric()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Perform a forward pass through the model `self.net`.

        :param x: A tensor of images.
        :return: A tensor of logits.
        """
        return self.model(x)

    def on_train_start(self) -> None:
        """Lightning hook that is called when training begins."""
        # by default lightning executes validation step sanity checks before training starts,
        # so it's worth to make sure validation metrics don't store results from these checks
        self.val_loss.reset()
        self.val_acc.reset()
        # self.val_acc_best.reset()

    def model_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Perform a single model step on a batch of data.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target labels.

        :return: A tuple containing (in order):
            - A tensor of losses.
            - A tensor of predictions.
            - A tensor of target labels.
        """

        preds = self.forward(batch)
        targets = batch["converged"]
        mask = batch["converged_mask"]
        loss = self.loss_fn(preds[mask], targets[mask].float())
        return loss, preds[mask], targets[mask]

    def training_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        """Perform a single training step on a batch of data from the training set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        :return: A tensor of losses between model predictions and targets.
        """
        loss, preds, targets = self.model_step(batch)

        # update and log metrics
        self.train_loss(loss)
        self.train_acc(preds, targets)
        self.log(
            "train/loss", self.train_loss, on_step=True, on_epoch=False, prog_bar=True
        )
        self.log("train/acc", self.train_acc, on_step=False, on_epoch=True, prog_bar=True)

        # return loss or backpropagation will fail
        return loss

    def on_train_epoch_end(self) -> None:
        "Lightning hook that is called when a training epoch ends."
        pass

    def validation_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> None:
        """Perform a single validation step on a batch of data from the validation set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        """
        loss, preds, targets = self.model_step(batch)

        # update and log metrics
        self.val_loss(loss)
        self.val_acc(preds, targets)

        self.log("val/loss", self.val_loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("val/acc", self.val_acc, on_step=True, on_epoch=True, prog_bar=True)

    def on_validation_epoch_end(self) -> None:
        "Lightning hook that is called when a validation epoch ends."
        # acc = self.val_acc.compute()  # get current val acc
        # self.val_acc_best(acc)  # update best so far val acc
        # log `val_acc_best` as a value through `.compute()` method, instead of as a metric object
        # otherwise metric would be reset by lightning after each epoch
        # self.log("val/acc_best", self.val_acc_best.compute(), sync_dist=True, prog_bar=True)
        pass

    def test_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> None:
        """Perform a single test step on a batch of data from the test set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        """
        loss, preds, targets = self.model_step(batch)

        # update and log metrics
        self.test_loss(loss)
        self.test_acc(preds, targets)

        self.log("test/loss", self.test_loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("test/acc", self.test_acc, on_step=True, on_epoch=True, prog_bar=True)
        # data_dict = self.trainer.datamodule.data_dict
        # print(batch)
        # print(len(batch["ion_key"]))
        # print(batch["asf_key"])
        # print(batch["asf_key"].shape)
        for i in range(len(batch["n_protons"])):
            ion_key = (batch["n_protons"][i].item(), batch["n_electrons"][i].item())
            excitations_tuple = tuple(map(tuple, batch["excitations"][i].cpu().numpy()))
            # print(ion_key, excitations_tuple)
            # print()
            # print(preds[i].cpu().numpy())
            if ion_key not in self.pred_dict:
                self.pred_dict[ion_key] = {}
            
            if excitations_tuple not in self.pred_dict[ion_key]:
                self.pred_dict[ion_key][excitations_tuple] = {}
            # Store the predictions using the excitations_tuple as the key
            # print()
            # print(preds.shape)
            # print(targets.shape)
            # print(preds[batch["converged_mask"]])
            # print(targets[batch["converged_mask"]])
            # print(batch["converged_mask"][i])
            # print()
            self.pred_dict[ion_key][excitations_tuple]['preds'] = copy.deepcopy(preds[i].cpu().numpy())
            self.pred_dict[ion_key][excitations_tuple]['targets'] = copy.deepcopy(targets[i].cpu().numpy())
            self.pred_dict[ion_key][excitations_tuple]['mask'] = copy.deepcopy(batch["converged_mask"][i].cpu().numpy())

            # ion_key = batch["ion_key"][i]  # Directly access the key without .item()
            # asf_key = batch["asf_key"][i]
            # # print(ion_key, asf_key)
            # data_dict[ion_key][asf_key]["predictions"] = preds[i].cpu().numpy()


    def on_test_epoch_end(self) -> None:
        """Lightning hook that is called when a test epoch ends."""
        
        with open("/Users/moustholmes/Projects/METAL-AI/data/data_dict_Li-Rh_converged_only_restults.pkl", 'wb') as f:
            pickle.dump(self.pred_dict, f)


    def setup(self, stage: str) -> None:
        """Lightning hook that is called at the beginning of fit (train + validate), validate,
        test, or predict.

        This is a good hook when you need to build models dynamically or adjust something about
        them. This hook is called on every process when using DDP.

        :param stage: Either `"fit"`, `"validate"`, `"test"`, or `"predict"`.
        """
        if self.hparams.compile and stage == "fit":
            self.net = torch.compile(self.net)

    def configure_optimizers(self) -> Dict[str, Any]:
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        Examples:
            https://lightning.ai/docs/pytorch/latest/common/lightning_module.html#configure-optimizers

        :return: A dict containing the configured optimizers and learning-rate schedulers to be used for training.
        """
        optimizer = self.hparams.optimizer(params=self.trainer.model.parameters())
        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val/loss",
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        return {"optimizer": optimizer}


if __name__ == "__main__":
    _ = MetalAILitModule(None, None, None, None)
