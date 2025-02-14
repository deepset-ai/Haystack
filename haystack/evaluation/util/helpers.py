# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from copy import deepcopy
from typing import Any, Dict, List


def aggregate_batched_pipeline_outputs(outputs: List[Dict[str, Dict[str, Any]]]) -> Dict[str, Dict[str, Any]]:
    """
    Combine the outputs of a pipeline that has been executed iteratively over a batch of inputs.

    Performs a transpose operation on the first and the third dimensions of the outputs.

    :param outputs:
        A list of outputs from the pipeline, where each output
        is a dictionary with the same keys and values with the
        same keys.
    :returns:
        The combined outputs.
    """
    # The pipeline is invoked iteratively over a batch of inputs, such
    # that each element in the outputs corresponds to a single element in
    # the batch input.
    if len(outputs) == 0:
        return {}
    if len(outputs) == 1:
        return outputs[0]

    # We'll use the first output as a sentinel to determine
    # if the shape of the rest of the outputs are the same.
    sentinel = outputs[0]
    for output in outputs[1:]:
        if output.keys() != sentinel.keys():
            raise ValueError(f"Expected components '{list(sentinel.keys())}' " f"but got '{list(output.keys())}'")

        for component_name, expected in sentinel.items():
            got = output[component_name]
            if got.keys() != expected.keys():
                raise ValueError(
                    f"Expected outputs from component '{component_name}' to have "
                    f"keys '{list(expected.keys())}' but got '{list(got.keys())}'"
                )

    # The outputs are of the correct/same shape. Now to transpose
    # the outermost list with the innermost dictionary.
    transposed: Dict[str, Dict[str, Any]] = {}
    for k, v in sentinel.items():
        transposed[k] = {k_h: [] for k_h in v.keys()}

    for output in outputs:
        for component_name, component_outputs in output.items():
            dest = transposed[component_name]
            for output_name, output_value in component_outputs.items():
                dest[output_name].append(output_value)

    return transposed


def deaggregate_batched_pipeline_inputs(inputs: Dict[str, Dict[str, List[Any]]]) -> List[Dict[str, Dict[str, Any]]]:
    """
    Separate the inputs of a pipeline that has been batched along its innermost dimension.

    Performs a transpose operation on the first and the third dimensions of the inputs.

    :param inputs:
        A dictionary of pipeline inputs that maps
        component-input pairs to lists of values.
    :returns:
        The separated inputs.
    """
    if len(inputs) == 0:
        return []

    # First component's inputs
    sentinel = next(iter(inputs.values()))
    # First component's first input's values
    sentinel = next(iter(sentinel.values()))  # type: ignore

    for component_name, component_inputs in inputs.items():
        for input_name, input_values in component_inputs.items():
            if len(input_values) != len(sentinel):
                raise ValueError(
                    f"Expected input '{component_name}.{input_name}' to have "
                    f"{len(sentinel)} values but got {len(input_values)}"
                )

    proto = {k: {k_h: None for k_h in v.keys()} for k, v in inputs.items()}
    transposed: List[Dict[str, Dict[str, Any]]] = []

    for i in range(len(sentinel)):
        new_dict = deepcopy(proto)
        for component_name, component_inputs in inputs.items():
            for input_name, input_values in component_inputs.items():
                new_dict[component_name][input_name] = input_values[i]
        transposed.append(new_dict)

    return transposed
