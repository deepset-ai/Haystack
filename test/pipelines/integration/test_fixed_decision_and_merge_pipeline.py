# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
import logging
from pathlib import Path
from pprint import pprint

from canals.pipeline import Pipeline
from sample_components import AddFixedValue, Parity, Double, Subtract

logging.basicConfig(level=logging.DEBUG)


def test_pipeline(tmp_path):
    pipeline = Pipeline()
    pipeline.add_component("add_one", AddFixedValue())
    pipeline.add_component("parity", Parity())
    pipeline.add_component("add_ten", AddFixedValue(add=10))
    pipeline.add_component("double", Double())
    pipeline.add_component("add_four", AddFixedValue(add=4))
    pipeline.add_component("add_two", AddFixedValue())
    pipeline.add_component("diff", Subtract())

    pipeline.connect("add_one.result", "parity.value")
    pipeline.connect("parity.even", "add_four.value")
    pipeline.connect("parity.odd", "double.value")
    pipeline.connect("add_ten.result", "diff.first_value")
    pipeline.connect("double.value", "diff.second_value")
    pipeline.connect("parity.odd", "add_ten.value")
    pipeline.connect("add_four.result", "add_two.value")

    # pipeline.draw(tmp_path / "fixed_decision_and_merge_pipeline.png")

    results = pipeline.run(
        {
            "add_one": {"value": 1},
            "add_two": {"add": 2},
        }
    )
    pprint(results)

    results == {
        "add_two": {"result": 8},
    }

    results = pipeline.run(
        {
            "add_one": {"value": 2},
            "add_two": {"add": 2},
        }
    )
    pprint(results)

    results == {
        "diff": {"difference": 7},
    }


if __name__ == "__main__":
    test_pipeline(Path(__file__).parent)
