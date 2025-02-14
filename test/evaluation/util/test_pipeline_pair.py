import pytest

from haystack import Pipeline
from haystack.evaluation.util.pipeline_pair import PipelinePair
from haystack.evaluation.util.helpers import aggregate_batched_pipeline_outputs

from haystack.testing.sample_components import AddFixedValue, Double, AddFixedValueBatch, DoubleBatch


@pytest.fixture
def first_pipeline():
    first = Pipeline()
    first.add_component("first_addition", AddFixedValue(add=10))
    first.add_component("second_addition", AddFixedValue(add=100))
    first.add_component("double", Double())
    first.connect("first_addition", "double")
    first.connect("double", "second_addition")
    return first


@pytest.fixture
def second_pipeline():
    second = Pipeline()
    second.add_component("first_addition", AddFixedValue(add=1))
    second.add_component("second_addition", AddFixedValue(add=2))
    second.add_component("double", Double())
    second.connect("first_addition", "double")
    second.connect("double", "second_addition")
    return second


@pytest.fixture
def second_pipeline_batched():
    second = Pipeline()
    second.add_component("first_addition", AddFixedValueBatch(add=1))
    second.add_component("second_addition", AddFixedValueBatch(add=2))
    second.add_component("double", DoubleBatch())
    second.connect("first_addition", "double")
    second.connect("double", "second_addition")
    return second


def test_pipeline_pair_init(first_pipeline, second_pipeline):
    pair = PipelinePair(
        first=first_pipeline,
        second=second_pipeline,
        outputs_to_inputs={"first_addition.result": ["first_addition.value"]},
    )


def test_pipeline_pair_invalid_io_specifier(first_pipeline, second_pipeline):
    with pytest.raises(ValueError, match="Invalid pipeline i/o path specifier"):
        _ = PipelinePair(
            first=first_pipeline, second=second_pipeline, outputs_to_inputs={"nonexistent": ["nonexistent"]}
        )


def test_pipeline_pair_invalid_first_component(first_pipeline, second_pipeline):
    with pytest.raises(ValueError, match="Output component .* not found in first pipeline."):
        _ = PipelinePair(
            first=first_pipeline, second=second_pipeline, outputs_to_inputs={"nonexistent.out": ["nonexistent.in"]}
        )


def test_pipeline_pair_invalid_first_component_output(first_pipeline, second_pipeline):
    with pytest.raises(ValueError, match="Component .* in first pipeline does not have expected output"):
        _ = PipelinePair(
            first=first_pipeline, second=second_pipeline, outputs_to_inputs={"double.out": ["nonexistent.in"]}
        )


def test_pipeline_pair_invalid_second_component(first_pipeline, second_pipeline):
    with pytest.raises(ValueError, match="Input component .* not found in second pipeline."):
        _ = PipelinePair(
            first=first_pipeline,
            second=second_pipeline,
            outputs_to_inputs={"first_addition.result": ["nonexistent.in"]},
        )


def test_pipeline_pair_invalid_second_component_input(first_pipeline, second_pipeline):
    with pytest.raises(ValueError, match="Component .* in second pipeline does not have expected input"):
        _ = PipelinePair(
            first=first_pipeline,
            second=second_pipeline,
            outputs_to_inputs={"first_addition.result": ["second_addition.some_input"]},
        )


def test_pipeline_pair_invalid_second_multiple_inputs(first_pipeline, second_pipeline):
    with pytest.raises(
        ValueError, match="Input .* in second pipeline is connected to multiple first pipeline outputs."
    ):
        _ = PipelinePair(
            first=first_pipeline,
            second=second_pipeline,
            outputs_to_inputs={
                "first_addition.result": ["second_addition.value"],
                "second_addition.result": ["second_addition.value"],
            },
        )


def test_pipeline_pair_run(first_pipeline, second_pipeline):
    pair = PipelinePair(
        first=first_pipeline,
        second=second_pipeline,
        outputs_to_inputs={"first_addition.result": ["first_addition.value"]},
        included_first_outputs={"first_addition"},
        included_second_outputs={"double"},
    )

    results = pair.run({"first_addition": {"value": 1}})
    assert results == {
        "first": {"first_addition": {"result": 11}, "second_addition": {"result": 122}},
        "second": {"double": {"value": 24}, "second_addition": {"result": 26}},
    }

    pair = PipelinePair(
        first=first_pipeline,
        second=second_pipeline,
        outputs_to_inputs={"first_addition.result": ["first_addition.value", "first_addition.add"]},
        included_first_outputs={"first_addition"},
        included_second_outputs={"first_addition", "double"},
    )

    results = pair.run({"first_addition": {"value": 10}})
    assert results == {
        "first": {"first_addition": {"result": 20}, "second_addition": {"result": 140}},
        "second": {"first_addition": {"result": 40}, "double": {"value": 80}, "second_addition": {"result": 82}},
    }


def test_pipeline_pair_run_second_extra_inputs(first_pipeline, second_pipeline):
    pair = PipelinePair(
        first=first_pipeline,
        second=second_pipeline,
        outputs_to_inputs={"first_addition.result": ["first_addition.value"]},
        included_first_outputs={"first_addition"},
        included_second_outputs={"first_addition", "double"},
    )

    results = pair.run(
        first_inputs={"first_addition": {"value": 1}},
        second_inputs={"first_addition": {"add": 10}, "second_addition": {"add": 100}},
    )
    assert results == {
        "first": {"first_addition": {"result": 11}, "second_addition": {"result": 122}},
        "second": {"first_addition": {"result": 21}, "double": {"value": 42}, "second_addition": {"result": 142}},
    }


def test_pipeline_pair_run_invalid_second_extra_inputs(first_pipeline, second_pipeline):
    pair = PipelinePair(
        first=first_pipeline,
        second=second_pipeline,
        outputs_to_inputs={"first_addition.result": ["first_addition.value"]},
        included_first_outputs={"first_addition"},
        included_second_outputs={"first_addition", "double"},
    )

    with pytest.raises(
        ValueError, match="Second pipeline input .* cannot be provided both explicitly and by the first pipeline"
    ):
        results = pair.run(
            first_inputs={"first_addition": {"value": 1}}, second_inputs={"first_addition": {"value": 10}}
        )


def test_pipeline_pair_run_map_first_outputs(first_pipeline, second_pipeline):
    pair = PipelinePair(
        first=first_pipeline,
        second=second_pipeline,
        outputs_to_inputs={"first_addition.result": ["first_addition.value"]},
        included_first_outputs={"first_addition"},
        included_second_outputs={"double"},
        map_first_outputs=lambda x: {"first_addition": {"result": 0}, "second_addition": {"result": 0}},
    )

    results = pair.run({"first_addition": {"value": 1}})
    assert results == {
        "first": {"first_addition": {"result": 0}, "second_addition": {"result": 0}},
        "second": {"double": {"value": 2}, "second_addition": {"result": 4}},
    }


def test_pipeline_pair_run_first_as_batch(first_pipeline, second_pipeline_batched):
    pair = PipelinePair(
        first=first_pipeline,
        second=second_pipeline_batched,
        outputs_to_inputs={"second_addition.result": ["first_addition.value"]},
        included_first_outputs={"first_addition"},
        included_second_outputs={"first_addition", "double"},
        map_first_outputs=lambda x: aggregate_batched_pipeline_outputs(x),
    )

    results = pair.run_first_as_batch([{"first_addition": {"value": i}} for i in range(5)])
    assert results == {
        "first": {
            "first_addition": {"result": [10, 11, 12, 13, 14]},
            "second_addition": {"result": [120, 122, 124, 126, 128]},
        },
        "second": {
            "first_addition": {"result": [121, 123, 125, 127, 129]},
            "double": {"value": [242, 246, 250, 254, 258]},
            "second_addition": {"result": [244, 248, 252, 256, 260]},
        },
    }


def test_pipeline_pair_run_first_as_batch_invalid_map_first_outputs(first_pipeline, second_pipeline_batched):
    pair = PipelinePair(
        first=first_pipeline,
        second=second_pipeline_batched,
        outputs_to_inputs={"second_addition.result": ["first_addition.value"]},
        included_first_outputs={"first_addition"},
        included_second_outputs={"first_addition", "double"},
        map_first_outputs=None,
    )

    with pytest.raises(ValueError, match="Mapping function for first pipeline outputs must be provided"):
        results = pair.run_first_as_batch([{"first_addition": {"value": i}} for i in range(5)])

    pair = PipelinePair(
        first=first_pipeline,
        second=second_pipeline_batched,
        outputs_to_inputs={"second_addition.result": ["first_addition.value"]},
        included_first_outputs={"first_addition"},
        included_second_outputs={"first_addition", "double"},
        map_first_outputs=lambda x: x,
    )

    with pytest.raises(ValueError, match="Mapping function must return an aggregate dictionary"):
        results = pair.run_first_as_batch([{"first_addition": {"value": i}} for i in range(5)])
