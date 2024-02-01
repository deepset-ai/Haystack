# SPDX-FileCopyrightText: 2023-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
from typing import List

from haystack.core.component import component
from haystack.core.pipeline import Pipeline
from haystack.testing.sample_components import StringListJoiner


@component
class InputMangler:
    @component.output_types(mangled_list=List[str])
    def run(self, input_list: List[str]):
        input_list.append("extra_item")
        return {"mangled_list": input_list}


def test_mutable_inputs():
    pipe = Pipeline()
    mangler1 = InputMangler()
    mangler2 = InputMangler()
    concat1 = StringListJoiner()
    concat2 = StringListJoiner()
    pipe.add_component("mangler1", mangler1)
    pipe.add_component("mangler2", mangler2)
    pipe.add_component("concat1", concat1)
    pipe.add_component("concat2", concat2)
    pipe.connect(mangler1.outputs.mangled_list, concat1.inputs.inputs)
    pipe.connect(mangler2.outputs.mangled_list, concat2.inputs.inputs)

    mylist = ["foo", "bar"]

    result = pipe.run(data={"mangler1": {"input_list": mylist}, "mangler2": {"input_list": mylist}})
    assert result["concat1"]["output"] == result["concat2"]["output"] == ["foo", "bar", "extra_item"]
