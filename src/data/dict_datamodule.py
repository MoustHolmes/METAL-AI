from typing import Any, Dict, Optional, Tuple

import torch
from lightning import LightningDataModule
from torch.utils.data import ConcatDataset, DataLoader, Dataset, random_split
from torchvision.datasets import MNIST
from torchvision.transforms import transforms

import pickle

from src.data.components.datasets import GroupedDictDataset
from src.data.components.samplers import GroupedBatchSampler
from src.data.components.collate_fns import dict_collate_fn

def split_data_dict(data_dict, validation_percentage=0.05):
    # Flatten the data_dict to a list of tuples: (ion_key, asf_key, excitations_length)
    flat_data = [
        (ion_key, asf_key, len(data_dict[ion_key][asf_key]["excitations"]))
        for ion_key in data_dict.keys()
        for asf_key in data_dict[ion_key].keys()
    ]

    # Sort by the length of excitations in descending order
    flat_data.sort(key=lambda x: x[2], reverse=True)

    # Determine the number of validation data points
    validation_size = int(len(flat_data) * validation_percentage)

    # Create the validation and training datasets
    validation_data = {}
    training_data = {}

    # Add the top 5% to the validation set
    for ion_key, asf_key, _ in flat_data[:validation_size]:
        if ion_key not in validation_data:
            validation_data[ion_key] = {}
        validation_data[ion_key][asf_key] = data_dict[ion_key][asf_key]

    # Add the remaining 95% to the training set
    for ion_key, asf_key, _ in flat_data[validation_size:]:
        if ion_key not in training_data:
            training_data[ion_key] = {}
        training_data[ion_key][asf_key] = data_dict[ion_key][asf_key]

    return training_data, validation_data

class DictDataModule(LightningDataModule):
    """`LightningDataModule` for the MNIST dataset.

    write about structure of the data, download, split, transform, etc...

    A `LightningDataModule` implements 7 key methods:

    ```python
        def prepare_data(self):
        # Things to do on 1 GPU/TPU (not on every GPU/TPU in DDP).
        # Download data, pre-process, split, save to disk, etc...

        def setup(self, stage):
        # Things to do on every process in DDP.
        # Load data, set variables, etc...

        def train_dataloader(self):
        # return train dataloader

        def val_dataloader(self):
        # return validation dataloader

        def test_dataloader(self):
        # return test dataloader

        def predict_dataloader(self):
        # return predict dataloader

        def teardown(self, stage):
        # Called on every process in DDP.
        # Clean up after fit or test.
    ```

    This allows you to share a full dataset without explaining how to download,
    split, transform and process the data.

    Read the docs:
        https://lightning.ai/docs/pytorch/latest/data/datamodule.html
    """

    def __init__(
        self,
        data_dir: str = "data/",
        train_val_test_split: Tuple[int, int, int] = (55_000, 5_000, 10_000),
        batch_size: int = 64,
        num_workers: int = 0,
        pin_memory: bool = False,
        persistent_workers: bool = False,
    ) -> None:
        """Initialize a `HDF5DataModule`.

        :param data_dir: The data directory. Defaults to `"data/"`.
        :param train_val_test_split: The train, validation and test split. Defaults to `(55_000, 5_000, 10_000)`.
        :param batch_size: The batch size. Defaults to `64`.
        :param num_workers: The number of workers. Defaults to `0`.
        :param pin_memory: Whether to pin memory. Defaults to `False`.
        """
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        self.data_dict = None

        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None

        self.batch_size_per_device = batch_size

    def prepare_data(self) -> None:
        """Download data if needed. Lightning ensures that `self.prepare_data()` is called only
        within a single process on CPU, so you can safely add your downloading logic within. In
        case of multi-node training, the execution of this hook depends upon
        `self.prepare_data_per_node()`.

        Do not use it to assign state (self.x = y).
        """
        with open(self.hparams.data_dir, "rb") as file:
            self.data_dict = pickle.load(file)
        self.data_dict_train, self.data_dict_val = split_data_dict(self.data_dict)

    def setup(self, stage: Optional[str] = None) -> None:
        """Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`.

        This method is called by Lightning before `trainer.fit()`, `trainer.validate()`, `trainer.test()`, and
        `trainer.predict()`, so be careful not to execute things like random split twice! Also, it is called after
        `self.prepare_data()` and there is a barrier in between which ensures that all the processes proceed to
        `self.setup()` once the data is prepared and available for use.

        :param stage: The stage to setup. Either `"fit"`, `"validate"`, `"test"`, or `"predict"`. Defaults to ``None``.
        """
        # Divide batch size by the number of devices.
        if self.trainer is not None:
            if self.hparams.batch_size % self.trainer.world_size != 0:
                raise RuntimeError(
                    f"Batch size ({self.hparams.batch_size}) is not divisible by the number of devices ({self.trainer.world_size})."
                )
            self.batch_size_per_device = (
                self.hparams.batch_size // self.trainer.world_size
            )
        
        # load and split datasets only if not loaded already
        if not self.data_train:
            self.data_train = GroupedDictDataset(data_dict=self.data_dict_train)
        if not self.data_val:
            self.data_val = GroupedDictDataset(data_dict=self.data_dict_val)
        if not self.data_test:
            self.data_test = GroupedDictDataset(data_dict=self.data_dict)

    def train_dataloader(self) -> DataLoader[Any]:
        """Create and return the train dataloader.

        :return: The train dataloader.
        """
        return DataLoader(
            dataset=self.data_train,
            batch_sampler=GroupedBatchSampler(self.data_train, batch_size =self.batch_size_per_device),
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            persistent_workers=self.hparams.persistent_workers,
        )

    def val_dataloader(self) -> DataLoader[Any]:
        """Create and return the validation dataloader.

        :return: The validation dataloader.
        """
        return DataLoader(
            dataset=self.data_val,
            batch_sampler=GroupedBatchSampler(self.data_val, batch_size =self.batch_size_per_device),
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            persistent_workers=self.hparams.persistent_workers,
        )

    def test_dataloader(self) -> DataLoader[Any]:
        """Create and return the test dataloader.

        :return: The test dataloader.
        """
        return DataLoader(
            dataset=self.data_test,
            batch_sampler=GroupedBatchSampler(self.data_test, batch_size =self.batch_size_per_device),
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            persistent_workers=self.hparams.persistent_workers,
        )

    def teardown(self, stage: Optional[str] = None) -> None:
        """Lightning hook for cleaning up after `trainer.fit()`, `trainer.validate()`,
        `trainer.test()`, and `trainer.predict()`.

        :param stage: The stage being torn down. Either `"fit"`, `"validate"`, `"test"`, or `"predict"`.
            Defaults to ``None``.
        """
        pass

    def state_dict(self) -> Dict[Any, Any]:
        """Called when saving a checkpoint. Implement to generate and save the datamodule state.

        :return: A dictionary containing the datamodule state that you want to save.
        """
        return {}

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Called when loading a checkpoint. Implement to reload datamodule state given datamodule
        `state_dict()`.

        :param state_dict: The datamodule state returned by `self.state_dict()`.
        """
        pass


if __name__ == "__main__":
    _ = DictDataModule()
