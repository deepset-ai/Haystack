# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

from tqdm import tqdm

from haystack import Pipeline


@dataclass(frozen=True)
class PipelinePair:  # pylint: disable=too-many-instance-attributes
    """
    A pair of pipelines that are linked together and executed sequentially.

    :param first:
        The first pipeline in the sequence.
    :param second:
        The second pipeline in the sequence.
    :param outputs_to_inputs:
        A mapping of the outputs of the first pipeline to the
        inputs of the second pipeline in the following format:
        `"name_of_component.name_of_output": "name_of_component.name_of_input`.
        A single output can be mapped to multiple inputs.
    :param map_first_outputs:
        A function that post-processes the outputs of the first
        pipeline, which it receives as its (only) argument.
    :param included_first_outputs:
        Names of components in the first pipeline whose outputs
        should be included in the final outputs.
    :param included_second_outputs:
        Names of components in the second pipeline whose outputs
        should be included in the final outputs.
    :param pre_execution_callback_first:
        A function that is called before the first pipeline is executed.
    :param pre_execution_callback_second:
        A function that is called before the second pipeline is executed.
    """

    first: Pipeline
    second: Pipeline
    outputs_to_inputs: Dict[str, List[str]]
    map_first_outputs: Optional[Callable] = None
    included_first_outputs: Optional[Set[str]] = None
    included_second_outputs: Optional[Set[str]] = None
    pre_execution_callback_first: Optional[Callable] = None
    pre_execution_callback_second: Optional[Callable] = None

    def __post_init__(self):
        first_outputs = self.first.outputs(include_components_with_connected_outputs=True)
        second_inputs = self.second.inputs(include_components_with_connected_inputs=True)
        seen_second_inputs = set()

        # Validate the mapping of outputs from the first pipeline
        # to the inputs of the second pipeline.
        for first_out, second_ins in self.outputs_to_inputs.items():
            first_comp_name, first_out_name = self._split_input_output_path(first_out)
            if first_comp_name not in first_outputs:
                raise ValueError(f"Output component '{first_comp_name}' not found in first pipeline.")
            if first_out_name not in first_outputs[first_comp_name]:
                raise ValueError(
                    f"Component '{first_comp_name}' in first pipeline does not have expected output '{first_out_name}'."
                )

            for second_in in second_ins:
                if second_in in seen_second_inputs:
                    raise ValueError(
                        f"Input '{second_in}' in second pipeline is connected to multiple first pipeline outputs."
                    )

                second_comp_name, second_input_name = self._split_input_output_path(second_in)
                if second_comp_name not in second_inputs:
                    raise ValueError(f"Input component '{second_comp_name}' not found in second pipeline.")
                if second_input_name not in second_inputs[second_comp_name]:
                    raise ValueError(
                        f"Component '{second_comp_name}' in second pipeline "
                        f"does not have expected input '{second_input_name}'."
                    )
                seen_second_inputs.add(second_in)

    def _validate_second_inputs(self, inputs: Dict[str, Dict[str, Any]]):
        # Check if the connected input is also provided explicitly.
        second_connected_inputs = [
            self._split_input_output_path(p_h) for p in self.outputs_to_inputs.values() for p_h in p
        ]
        for component_name, input_name in second_connected_inputs:
            provided_input = inputs.get(component_name)
            if provided_input is None:
                continue
            if input_name in provided_input:
                raise ValueError(
                    f"Second pipeline input '{component_name}.{input_name}' cannot "
                    "be provided both explicitly and by the first pipeline."
                )

    @staticmethod
    def _split_input_output_path(path: str) -> Tuple[str, str]:
        # Split the input/output path into component name and input/output name.
        pos = path.find(".")
        if pos == -1:
            raise ValueError(
                f"Invalid pipeline i/o path specifier '{path}' - Must be "
                "in the following format: <component_name>.<input/output_name>"
            )
        return path[:pos], path[pos + 1 :]

    def _prepare_required_outputs_for_first_pipeline(self) -> Set[str]:
        # To ensure that we have all the outputs from the first
        # pipeline that are required by the second pipeline, we
        # collect first collect all the keys in the first-to-second
        # output-to-input mapping and then add the explicitly included
        # first pipeline outputs.
        first_components_with_outputs = {self._split_input_output_path(p)[0] for p in self.outputs_to_inputs.keys()}
        if self.included_first_outputs is not None:
            first_components_with_outputs = first_components_with_outputs.union(self.included_first_outputs)
        return first_components_with_outputs

    def _map_first_second_pipeline_io(
        self, first_outputs: Dict[str, Dict[str, Any]], second_inputs: Dict[str, Dict[str, Any]]
    ) -> Dict[str, Dict[str, Any]]:
        # Map the first pipeline outputs to the second pipeline inputs.
        for first_output, second_input_candidates in self.outputs_to_inputs.items():
            first_component, first_output = self._split_input_output_path(first_output)

            # Each output from the first pipeline can be mapped to multiple inputs in the second pipeline.
            for second_input in second_input_candidates:
                second_component, second_input_socket = self._split_input_output_path(second_input)

                second_component_inputs = second_inputs.get(second_component)
                if second_component_inputs is not None:
                    # Pre-condition should've been validated earlier.
                    assert second_input_socket not in second_component_inputs
                    # The first pipeline's output should also guaranteed at this point.
                    second_component_inputs[second_input_socket] = first_outputs[first_component][first_output]
                else:
                    second_inputs[second_component] = {
                        second_input_socket: first_outputs[first_component][first_output]
                    }

        return second_inputs

    def run(
        self, first_inputs: Dict[str, Dict[str, Any]], second_inputs: Optional[Dict[str, Dict[str, Any]]] = None
    ) -> Dict[str, Dict[str, Any]]:
        """
        Execute the pipeline pair in sequence.

        Invokes the first pipeline and then the second with the outputs
        of the former. This assumes that both pipelines have the same input
        modality, i.e., the shapes of the first pipeline's outputs match the
        shapes of the second pipeline's inputs.

        :param first_inputs:
            The inputs to the first pipeline.
        :param second_inputs:
            The inputs to the second pipeline.
        :returns:
            A dictionary with the following keys:
            - `first` - The outputs of the first pipeline.
            - `second` - The outputs of the second pipeline.
        """
        second_inputs = second_inputs or {}
        self._validate_second_inputs(second_inputs)

        if self.pre_execution_callback_first is not None:
            self.pre_execution_callback_first()
        first_outputs = self.first.run(
            first_inputs, include_outputs_from=self._prepare_required_outputs_for_first_pipeline()
        )
        if self.map_first_outputs is not None:
            first_outputs = self.map_first_outputs(first_outputs)
        second_inputs = self._map_first_second_pipeline_io(first_outputs, second_inputs)

        if self.pre_execution_callback_second is not None:
            self.pre_execution_callback_second()
        second_outputs = self.second.run(second_inputs, include_outputs_from=self.included_second_outputs)

        return {"first": first_outputs, "second": second_outputs}

    def run_first_as_batch(
        self,
        first_inputs: List[Dict[str, Dict[str, Any]]],
        second_inputs: Optional[Dict[str, Dict[str, Any]]] = None,
        *,
        progress_bar: bool = False,
    ) -> Dict[str, Dict[str, Any]]:
        """
        Execute the pipeline pair in sequence.

        Invokes the first pipeline iteratively over the list of inputs and
        passing the cumulative outputs to the second pipeline. This is suitable
        when the first pipeline has a single logical input-to-output mapping and the
        second pipeline expects multiple logical inputs, e.g: a retrieval
        pipeline that accepts a single query and returns a list of documents
        and an evaluation pipeline that accepts multiple lists of documents
        and multiple lists of ground truth data.

        :param first_inputs:
            A batch of inputs to the first pipeline. A mapping
            function must be provided to aggregate the outputs.
        :param second_inputs:
            The inputs to the second pipeline.
        :param progress_bar:
            Whether to display a progress bar for the execution
            of the first pipeline.
        :returns:
            A dictionary with the following keys:
            - `first` - The (aggregate) outputs of the first pipeline.
            - `second` - The outputs of the second pipeline.
        """
        second_inputs = second_inputs or {}
        self._validate_second_inputs(second_inputs)

        first_components_with_outputs = self._prepare_required_outputs_for_first_pipeline()
        if self.map_first_outputs is None:
            raise ValueError("Mapping function for first pipeline outputs must be provided for batch execution.")

        if self.pre_execution_callback_first is not None:
            self.pre_execution_callback_first()
        first_outputs: Dict[str, Dict[str, Any]] = self.map_first_outputs(
            [
                self.first.run(i, include_outputs_from=first_components_with_outputs)
                for i in tqdm(first_inputs, disable=not progress_bar)
            ]
        )
        if not isinstance(first_outputs, dict):
            raise ValueError("Mapping function must return an aggregate dictionary of outputs.")

        second_inputs = self._map_first_second_pipeline_io(first_outputs, second_inputs)

        if self.pre_execution_callback_second is not None:
            self.pre_execution_callback_second()
        second_outputs = self.second.run(second_inputs, include_outputs_from=self.included_second_outputs)

        return {"first": first_outputs, "second": second_outputs}
