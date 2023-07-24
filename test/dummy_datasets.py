# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
from typing import Callable, Optional

import torch
from torch.utils.data import Dataset

from renate import defaults
from renate.data.data_module import RenateDataModule
from renate.utils.pytorch import get_generator


class DummyDataset(Dataset):
    def __init__(self, data, targets, transform=None):
        assert len(data) == len(targets)
        self.data = data
        self.targets = targets
        self.transform = transform

    def __getitem__(self, index):
        x = self.data[index]
        y = self.targets[index]
        if self.transform is not None:
            x = self.transform(x)
        return x, y

    def __len__(self):
        return len(self.data)


class DummyTorchVisionDataModule(RenateDataModule):
    """
    A simple data module similar to `TorchVisionDataModule` with 100 training instances and 100 test
    instances with shape (1, 5, 5) and 5 classes.
    """

    def __init__(
        self,
        transform: Optional[Callable] = None,
        val_size: float = defaults.VALIDATION_SIZE,
        seed: int = defaults.SEED,
    ):
        super().__init__(
            data_path="",
            src_bucket="",
            src_object_name="",
            val_size=val_size,
            seed=seed,
        )
        self._transform = transform
        self.input_shape = (1, 5, 5)
        self.output_shape = (1,)
        self.output_classes = 5

    def prepare_data(self) -> None:
        pass

    def _get_random_data(self, offset: int = 0):
        rng = get_generator(self._seed + offset)
        X = torch.randint(0, 42, (100, *self.input_shape), generator=rng, dtype=torch.float32)
        y = torch.ones(100, dtype=torch.long)
        for i in range(5):
            y[i * 20 : (i + 1) * 20] = i
        return X, y

    def setup(self):
        self.X_train, self.y_train = self._get_random_data()
        train_data = DummyDataset(self.X_train, self.y_train, self._transform)
        self._train_data, self._val_data = self._split_train_val_data(train_data)
        self.X_test, self.y_test = self._get_random_data()
        self._test_data = DummyDataset(self.X_test, self.y_test, self._transform)
