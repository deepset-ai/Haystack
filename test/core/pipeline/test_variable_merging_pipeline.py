# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
import logging

from haystack.core.pipeline import Pipeline
from haystack.testing.sample_components import AddFixedValue, Sum

logging.basicConfig(level=logging.DEBUG)


def test_pipeline():
    pipeline = Pipeline()
    first_addition = AddFixedValue(add=2)
    second_addition = AddFixedValue(add=2)
    third_addition = AddFixedValue(add=2)
    sum = Sum()
    fourth_addition = AddFixedValue(add=1)
    pipeline.add_component("first_addition", first_addition)
    pipeline.add_component("second_addition", second_addition)
    pipeline.add_component("third_addition", third_addition)
    pipeline.add_component("sum", sum)
    pipeline.add_component("fourth_addition", fourth_addition)

    pipeline.connect(first_addition.outputs.result, second_addition.inputs.value)
    pipeline.connect(first_addition.outputs.result, sum.inputs.values)
    pipeline.connect(second_addition.outputs.result, sum.inputs.values)
    pipeline.connect(third_addition.outputs.result, sum.inputs.values)
    pipeline.connect(sum.outputs.total, fourth_addition.inputs.value)

    results = pipeline.run({"first_addition": {"value": 1}, "third_addition": {"value": 1}})
    assert results == {"fourth_addition": {"result": 12}}
