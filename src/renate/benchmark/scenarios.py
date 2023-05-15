# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
import abc
from typing import Callable, List, Tuple, Union

import numpy as np
import torch
from torch.utils.data import Dataset, Subset
from torchvision.transforms import Lambda, RandomRotation, ToPILImage

from renate import defaults
from renate.benchmark.datasets.wild_time_data import WildTimeDataModule
from renate.data.data_module import RenateDataModule
from renate.data.datasets import _TransformedDataset
from renate.utils.pytorch import get_generator, randomly_split_data


class Scenario(abc.ABC):
    """Creates a continual learning scenario from a RenateDataModule.

    This class can be extended to modify the returned training/validation/test sets
    to implement different experimentation settings.

    Note that many scenarios implemented here perform randomized operations, e.g., to split a base
    dataset into chunks. The scenario is only reproducible if the _same_ seed is provided in
    subsequent instantiations. The seed argument is required for these scenarios.

    Args:
        data_module: The source RenateDataModule for the the user data.
        num_tasks: The total number of expected tasks for experimentation.
        chunk_id: The data chunk to load in for the training or validation data.
        seed: Seed used to fix random number generation.
    """

    def __init__(
        self,
        data_module: RenateDataModule,
        num_tasks: int,
        chunk_id: int,
        seed: int = defaults.SEED,
    ) -> None:
        super().__init__()
        self._data_module = data_module
        self._num_tasks = num_tasks
        self._verify_chunk_id(chunk_id)
        self._chunk_id = chunk_id
        self._seed = seed
        self._train_data: Dataset = None
        self._val_data: Dataset = None
        self._test_data: List[Dataset] = None

    def prepare_data(self) -> None:
        """Downloads datasets."""
        self._data_module.prepare_data()

    @abc.abstractmethod
    def setup(self) -> None:
        """Sets up the scenario."""
        pass

    def train_data(self) -> Dataset:
        """Returns training dataset with respect to current `chunk_id`."""
        return self._train_data

    def val_data(self) -> Dataset:
        """Returns validation dataset with respect to current `chunk_id`."""
        return self._val_data

    def test_data(self) -> List[Dataset]:
        """Returns the test data with respect to all tasks in `num_tasks`."""
        return self._test_data

    def _verify_chunk_id(self, chunk_id: int) -> None:
        """A helper function to verify that the `chunk_id` is valid."""
        assert 0 <= chunk_id < self._num_tasks

    def _split_and_assign_train_and_val_data(self) -> None:
        """Performs train/val split and assigns the `train_data` and `val_data` attributes."""
        proportions = [1 / self._num_tasks for _ in range(self._num_tasks)]
        train_data = self._data_module.train_data()
        self._train_data = randomly_split_data(train_data, proportions, self._seed)[self._chunk_id]
        if self._data_module.val_data():
            val_data = self._data_module.val_data()
            self._val_data = randomly_split_data(val_data, proportions, self._seed)[self._chunk_id]


class BenchmarkScenario(Scenario):
    """This is a scenario to concatenate test data of a data module, which by definition has
    different chunks.
    """

    def setup(self) -> None:
        self._data_module.setup()
        self._train_data = self._data_module.train_data()
        self._val_data = self._data_module.val_data()
        self._test_data = self._data_module._test_data


class ClassIncrementalScenario(Scenario):
    """A scenario that creates data chunks from data samples with specific classes from a data
    module.

    This class, upon giving a list describing the separation of the dataset separates the dataset
    with respect to classification labels.

    Note that, in order to apply this scenario, the scenario assumes that the data points in the
    data module are organised into tuples of exactly 2 tensors i.e. `(x, y)` where `x` is the input
    and `y` is the class id.

    Args:
        data_module: The source RenateDataModule for the the user data.
        chunk_id: The data chunk to load in for the training or validation data.
        class_groupings: List of lists, describing the division of the classes for respective tasks.
    """

    def __init__(
        self,
        data_module: RenateDataModule,
        chunk_id: int,
        class_groupings: List[List[int]],
    ) -> None:
        super().__init__(data_module, len(class_groupings), chunk_id)
        self._class_groupings = class_groupings

    def setup(self) -> None:
        """Make assignments: val/train/test splits."""
        self._data_module.setup()
        self._train_data = self._get_task_subset(
            self._data_module.train_data(), chunk_id=self._chunk_id
        )
        if self._data_module.val_data():
            self._val_data = self._get_task_subset(
                self._data_module.val_data(), chunk_id=self._chunk_id
            )
        self._test_data = [
            self._get_task_subset(self._data_module.test_data(), i) for i in range(self._num_tasks)
        ]

    def _get_task_subset(self, dataset: Dataset, chunk_id: int) -> Dataset:
        """A helper function identifying indices corresponding to given classes."""
        class_group = self._class_groupings[chunk_id]
        indices = torch.tensor(
            [i for i in range(len(dataset)) if dataset[i][1] in class_group],
            dtype=torch.long,
        )
        subset = Subset(dataset, indices)
        targets = set(np.unique([subset[i][1].item() for i in range(len(subset))]))
        expected_targets = set(self._class_groupings[chunk_id])
        if targets != expected_targets:
            raise ValueError(
                f"Chunk {chunk_id} does not contain classes "
                f"{sorted(list(expected_targets - targets))}."
            )
        return subset


class TransformScenario(Scenario):
    """A scenario that applies a different transformation to each chunk.

    The base ``data_module`` is split into ``len(transforms)`` random chunks. Then ``transforms[i]``
    is applied to chunk ``i``.

    Args:
        data_module: The base data module.
        transforms: A list of transformations.
        chunk_id: The id of the chunk to retrieve.
        seed: Seed used to fix random number generation.
    """

    def __init__(
        self,
        data_module: RenateDataModule,
        transforms: List[Callable],
        chunk_id: int,
        seed: int = defaults.SEED,
    ) -> None:
        super().__init__(data_module, len(transforms), chunk_id, seed)
        self._transforms = transforms

    def setup(self) -> None:
        self._data_module.setup()
        self._split_and_assign_train_and_val_data()
        self._train_data = _TransformedDataset(
            self._train_data, transform=self._transforms[self._chunk_id]
        )
        if self._val_data:
            self._val_data = _TransformedDataset(
                self._val_data, transform=self._transforms[self._chunk_id]
            )
        self._test_data = []
        for i in range(self._num_tasks):
            self._test_data.append(
                _TransformedDataset(self._data_module.test_data(), transform=self._transforms[i])
            )


class ImageRotationScenario(TransformScenario):
    """A scenario that rotates the images in the dataset by a different angle for each chunk.

    Args:
        data_module: The base data module.
        degrees: List of degrees corresponding to different tasks.
        chunk_id: The data chunk to load in for the training or validation data.
        seed: Seed used to fix random number generation.
    """

    def __init__(
        self,
        data_module: RenateDataModule,
        degrees: List[int],
        chunk_id: int,
        seed: int,
    ) -> None:
        transforms = [RandomRotation(degrees=(deg, deg)) for deg in degrees]
        super().__init__(data_module, transforms, chunk_id, seed)


class PermutationScenario(TransformScenario):
    """A scenario that applies a different random permutation of features for each chunk.

    Args:
        data_module: The base data module.
        num_tasks: The total number of expected tasks for experimentation.
        input_dim: Dimension of the inputs. Can be a shape tuple or the total number of features.
        chunk_id: The data chunk to load in for the training or validation data.
        seed: A random seed to fix the random number generation for permutations.
    """

    def __init__(
        self,
        data_module: RenateDataModule,
        num_tasks: int,
        input_dim: Union[List[int], Tuple[int], int],
        chunk_id: int,
        seed: int,
    ) -> None:
        input_dim = np.prod(input_dim)
        rng = get_generator(seed)
        transforms = [torch.nn.Identity()]
        for _ in range(num_tasks - 1):
            permutation = torch.randperm(input_dim, generator=rng)
            transform = Lambda(lambda x, p=permutation: x.flatten()[p].view(x.size()))
            transforms.append(transform)
        super().__init__(data_module, transforms, chunk_id, seed)


class IIDScenario(Scenario):
    """A scenario splitting datasets into random equally-sized chunks."""

    def setup(self) -> None:
        """Make assignments: val/train/test splits."""
        self._data_module.setup()
        proportions = [1 / self._num_tasks for _ in range(self._num_tasks)]
        self._train_data = randomly_split_data(
            self._data_module.train_data(), proportions, self._seed
        )[self._chunk_id]
        val_data = self._data_module.val_data()
        if val_data:
            self._val_data = randomly_split_data(val_data, proportions, self._seed)[self._chunk_id]
        self._test_data = [self._data_module.test_data() for _ in range(self._num_tasks)]


class _SortingScenario(Scenario):
    """A scenario that _softly_ sorts a dataset by some score.

    Randomness in the sorted order is induced by swapping the position of random pairs.

    Args:
        data_module: The source RenateDataModule for the the user data.
        num_tasks: The total number of expected tasks for experimentation.
        randomness: A value between 0 and 1. For a dataset with ``N`` data points,
            ``0.5 * N * randomness`` random pairs are swapped.
        chunk_id: The data chunk to load in for the training or validation data.
        seed: Seed used to fix random number generation.
    """

    def __init__(
        self,
        data_module: RenateDataModule,
        num_tasks: int,
        randomness: float,
        chunk_id: int,
        seed: int = defaults.SEED,
    ) -> None:
        super().__init__(data_module, num_tasks, chunk_id, seed)
        self._randomness = randomness
        assert 0 <= self._randomness <= 1

    @abc.abstractmethod
    def _get_scores(self, dataset: Dataset) -> List[float]:
        """Returns a float value for each data point which is used for sorting."""
        pass

    def _split(self, dataset: Dataset) -> List[Dataset]:
        """Sorts data, applies random swapping and then returns the Datasets."""
        scores = self._get_scores(dataset)
        idx_ordered = [x for _, x in sorted(zip(scores, np.arange(len(dataset))))]
        rng = np.random.RandomState(seed=self._seed)
        for _ in range(int(self._randomness * len(dataset) / 2)):
            i, j = rng.randint(len(dataset), size=2)
            idx_ordered[i], idx_ordered[j] = idx_ordered[j], idx_ordered[i]
        split = torch.tensor_split(torch.tensor(idx_ordered), self._num_tasks)
        return [Subset(dataset, idx) for idx in split]

    def setup(self) -> None:
        """Make assignments: val/train/test splits."""
        self._data_module.setup()
        train_data = self._data_module.train_data()
        self._train_data = self._split(train_data)[self._chunk_id]
        val_data = self._data_module.val_data()
        if val_data:
            self._val_data = self._split(val_data)[self._chunk_id]
        test_data = self._data_module.test_data()
        self._test_data = self._split(test_data)


class FeatureSortingScenario(_SortingScenario):
    """A scenario that _softly_ sorts a dataset by the value of a feature, then creates chunks.

    This scenario sorts the data according to a feature value (see `feature_idx`) and randomly
    swaps data positions based on the degree of randomness (see `randomness`).

    This scenario assumes that `dataset[i]` returns a tuple `(x, y)` with a tensor `x` containing
    the features.

    Args:
        data_module: The source RenateDataModule for the the user data.
        num_tasks: The total number of expected tasks for experimentation.
        feature_idx: Index of the feature by which to sort. This index refers to the input features
            `x` of a single data point, i.e., no batch dimension. If the tensor `x` has more than
            one dimension, this indexes along the 0-dim while additional dimensions will be averaged
            out. Hence, for images, `feature_idx` refers to a color channel and we sort by mean
            color channel value.
        randomness: A value between 0 and 1. For a dataset with ``N`` data points,
            ``0.5 * N * randomness`` random pairs are swapped.
        chunk_id: The data chunk to load in for the training or validation data.
        seed: Seed used to fix random number generation.
    """

    def __init__(
        self,
        data_module: RenateDataModule,
        num_tasks: int,
        feature_idx: int,
        randomness: float,
        chunk_id: int,
        seed: int = defaults.SEED,
    ) -> None:
        super().__init__(
            data_module=data_module,
            num_tasks=num_tasks,
            randomness=randomness,
            chunk_id=chunk_id,
            seed=seed,
        )
        self._feature_idx = feature_idx

    def _get_scores(self, dataset: Dataset) -> List[float]:
        return [x[0][self._feature_idx].mean().item() for x, _ in dataset]


class HueShiftScenario(_SortingScenario):
    """A scenario that sorts an image dataset by the hue value, then creates chunks.

    All images are sorted by hue value and divided into ``num_tasks`` tasks.
    ``randomness`` is a value between 0 and 1 and controls the number of random swaps applied
    to the sorting.

    This scenario assumes that `dataset[i]` returns a tuple `(x, y)` with a tensor `x` containing
    an RGB image.
    """

    def _get_scores(self, dataset: Dataset) -> List[float]:
        scores = []
        to_pil_image = ToPILImage()
        for image, _ in dataset:
            count, value = np.histogram(
                np.array(to_pil_image(image).convert("HSV"))[:, :, 0].reshape(-1), bins=100
            )
            scores.append(value[np.argmax(count)])
        return scores


class WildTimeScenario(Scenario):
    """Creating a time-incremental scenario for the Wild-Time datasets.

    In contrast to the original work, data is presented time step by time step (no grouping) and
    the test set is all data up to the current time step.

    Args:
        data_module: The source RenateDataModule for the the user data.
        num_tasks: The total number of expected tasks for experimentation.
        chunk_id: The data chunk to load in for the training or validation data.
        seed: Seed used to fix random number generation.
    """

    def __init__(
        self,
        data_module: RenateDataModule,
        num_tasks: int,
        chunk_id: int,
        seed: int = defaults.SEED,
    ) -> None:
        super().__init__(data_module=data_module, num_tasks=num_tasks, chunk_id=chunk_id, seed=seed)
        if not isinstance(data_module, WildTimeDataModule):
            raise ValueError("This scenario is only compatible with `WildTimeDataModule`.")

    def setup(self) -> None:
        """Sets up the scenario."""
        self._data_module.time_step = self._chunk_id
        self._data_module.setup()
        self._train_data = self._data_module.train_data()
        self._val_data = self._data_module.val_data()
        self._test_data = []
        for i in range(self._num_tasks):
            self._data_module.time_step = i
            self._data_module.setup()
            self._test_data.append(self._data_module.test_data())
