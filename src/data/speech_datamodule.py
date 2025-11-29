import pickle
from typing import Any, Dict, Optional, Tuple

from hydra.utils import to_absolute_path
from lightning import LightningDataModule
import torch
from torch.utils.data import DataLoader, Dataset

from src.utils.augmentations import build_augmentation_pipeline
from src.data.components.speech_dataset import SpeechDataset, speech_collate_fn


class SpeechDataModule(LightningDataModule):
    """`LightningDataModule` that wraps the speech neural decoding dataset.
    
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
        # define default params for data module
        self,
        dataset_path: str,
        batch_size: int = 16,
        num_workers: int = 0,
        pin_memory: bool = False,
        shuffle: bool = True,
        persistent_workers: bool = False,
        seq_len: int = 150,
        max_time_series_len: int = 1200,
        seed: int = 0,
        white_noise_std: float = 0.0,
        constant_offset_std: float = 0.0,
        n_classes: Optional[int] = None,
        n_input_features: Optional[int] = None,
        n_days: Optional[int] = None,
    ) -> None:
        super().__init__()

        self.dataset_path = dataset_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.shuffle = shuffle
        self.persistent_workers = persistent_workers
        self.seq_len = seq_len
        self.max_time_series_len = max_time_series_len
        self.seed = seed
        self.n_classes = n_classes
        self.n_input_features = n_input_features
        self.n_days = n_days

        self.transforms = build_augmentation_pipeline(white_noise_std, constant_offset_std)

        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None
        self.n_input_features: Optional[int] = n_input_features

    def prepare_data(self) -> None:
        # nothing to download; data expected to be present locally
        return

    def _load_dataset(self) -> Dict[str, Any]:
        with open(to_absolute_path(self.dataset_path), "rb") as handle:
            return pickle.load(handle)

    def setup(self, stage: Optional[str] = None) -> None:
        """Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`.

        This method is called by Lightning before `trainer.fit()`, `trainer.validate()`, `trainer.test()`, and
        `trainer.predict()`, so be careful not to execute things like random split twice! Also, it is called after
        `self.prepare_data()` and there is a barrier in between which ensures that all the processes proceed to
        `self.setup()` once the data is prepared and available for use.

        :param stage: The stage to setup. Either `"fit"`, `"validate"`, `"test"`, or `"predict"`. Defaults to ``None``.
        """
        if self.data_train is not None:
            return

        data = self._load_dataset()

        self.data_train = SpeechDataset(data["train"], transform=self.transforms)
        self.data_val = SpeechDataset(data["test"])
        self.data_test = SpeechDataset(data["test"]) # the actual test set is only available during evaluation phase

        self.n_days = self.n_days or self.data_train.n_days

        # infer dimensions if they were not provided explicitly
        example_features = self.data_train.neural_feats[0]
        if self.n_input_features is None:
            self.n_input_features = int(example_features.shape[1])
        if self.n_classes is None:
            max_label = 0
            for labels in self.data_train.phone_seqs:
                max_label = max(max_label, int(torch.as_tensor(labels).max().item()))
            # labels are assumed to reserve index 0 for the CTC blank token
            self.n_classes = max_label

    def train_dataloader(self) -> DataLoader[Any]:
        """Create and return the train dataloader.

        :return: The train dataloader.
        """
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers if self.num_workers > 0 else False,
            shuffle=self.shuffle,
            collate_fn=speech_collate_fn,
        )

    def val_dataloader(self) -> DataLoader[Any]:
        """Create and return the validation dataloader.

        :return: The validation dataloader.
        """
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers if self.num_workers > 0 else False,
            shuffle=False,
            collate_fn=speech_collate_fn,
        )

    def test_dataloader(self) -> DataLoader[Any]:
        """Create and return the test dataloader.

        :return: The test dataloader.
        """
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers if self.num_workers > 0 else False,
            shuffle=False,
            collate_fn=speech_collate_fn,
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
    _ = SpeechDataModule()
