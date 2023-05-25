# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
from typing import Any, Dict, Tuple, Type

import pytest
from conftest import LEARNER_KWARGS, LEARNERS

from renate.models import RenateModule
from renate.updaters.learner import Learner


def get_model_and_learner_and_learner_kwargs(
    learner_class: Type[Learner],
) -> Tuple[RenateModule, Learner, Dict[str, Any]]:
    learner_kwargs = LEARNER_KWARGS[learner_class]
    model = pytest.helpers.get_renate_module_mlp(
        num_inputs=1, num_outputs=1, hidden_size=1, num_hidden_layers=1
    )
    learner = learner_class(model=model, **learner_kwargs)
    return model, learner, learner_kwargs


def check_learner_variables(learner: Learner, expected_variable_values: Dict[str, Any]):
    for attribute_name, attribute_value in expected_variable_values.items():
        if attribute_name in [
            "memory_size",
            "learner_class_name",
            "val_memory_buffer",
            "memory_buffer",
        ]:
            continue
        assert getattr(learner, f"_{attribute_name}") == attribute_value


@pytest.mark.parametrize("learner_class", LEARNERS)
def test_save_and_load_learner(tmpdir, learner_class):
    model, learner, learner_kwargs = get_model_and_learner_and_learner_kwargs(learner_class)
    checkpoint_dict = {}
    learner.on_save_checkpoint(checkpoint=checkpoint_dict)
    check_learner_variables(learner, checkpoint_dict)
