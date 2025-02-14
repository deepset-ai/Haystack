# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
import pytest

from haystack.evaluation.util.helpers import aggregate_batched_pipeline_outputs, deaggregate_batched_pipeline_inputs


def test_aggregate_batched_pipeline_outputs_empty():
    assert aggregate_batched_pipeline_outputs([]) == {}


def test_aggregate_batched_pipeline_outputs_single():
    assert aggregate_batched_pipeline_outputs([{"a": {"b": [1, 2]}}]) == {"a": {"b": [1, 2]}}


def test_aggregate_batched_pipeline_outputs_multiple():
    outputs = [{"a": {"b": [1, 2], "c": [10, 20]}}, {"a": {"b": [3, 4], "c": [30, 40]}}]
    assert aggregate_batched_pipeline_outputs(outputs) == {"a": {"b": [[1, 2], [3, 4]], "c": [[10, 20], [30, 40]]}}


def test_aggregate_batched_pipeline_outputs_mismatched_components():
    outputs = [{"a": {"b": [1, 2]}}, {"c": {"b": [3, 4]}}]
    with pytest.raises(ValueError, match="Expected components .* but got"):
        aggregate_batched_pipeline_outputs(outputs)


def test_aggregate_batched_pipeline_outputs_mismatched_component_outputs():
    outputs = [{"a": {"b": [1, 2]}}, {"a": {"b": [3, 4], "c": [5, 6]}}]
    with pytest.raises(ValueError, match="Expected outputs from component .* to have keys .* but got"):
        aggregate_batched_pipeline_outputs(outputs)


def test_deaggregate_batched_pipeline_inputs_empty():
    assert deaggregate_batched_pipeline_inputs({}) == []


def test_deaggregate_batched_pipeline_inputs_single():
    inputs = {"a": {"b": [1, 2]}}
    assert deaggregate_batched_pipeline_inputs(inputs) == [{"a": {"b": 1}}, {"a": {"b": 2}}]


def test_deaggregate_batched_pipeline_inputs_multiple():
    inputs = {"a": {"b": [1, 2], "c": [10, 20]}}
    assert deaggregate_batched_pipeline_inputs(inputs) == [{"a": {"b": 1, "c": 10}}, {"a": {"b": 2, "c": 20}}]


def test_deaggregate_batched_pipeline_inputs_shape_mismatch():
    inputs = {"a": {"b": [1, 2]}, "c": {"b": [3]}}
    with pytest.raises(ValueError, match="Expected input .* to have *. values but got"):
        deaggregate_batched_pipeline_inputs(inputs)
