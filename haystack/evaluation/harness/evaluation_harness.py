# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, Generic, Optional, Type, TypeVar

from haystack import Pipeline
from haystack.core.serialization import DeserializationCallbacks


@dataclass
class EvaluationRunOverrides:
    """
    Overrides for an evaluation run.

    Use it to override the init parameters of components in either
    or both the evaluated and evaluation pipelines. Each key is
    a component name, and its value is a dictionary with init parameters
    to override.

    :param evaluated_pipeline_overrides:
        Overrides for the evaluated pipeline.
    :param evaluation_pipeline_overrides:
        Overrides for the evaluation pipeline.
    """

    evaluated_pipeline_overrides: Optional[Dict[str, Dict[str, Any]]] = None
    evaluation_pipeline_overrides: Optional[Dict[str, Dict[str, Any]]] = None


EvalRunInputT = TypeVar("EvalRunInputT")
EvalRunOutputT = TypeVar("EvalRunOutputT")
EvalRunOverridesT = TypeVar("EvalRunOverridesT")


class EvaluationHarness(ABC, Generic[EvalRunInputT, EvalRunOverridesT, EvalRunOutputT]):
    """
    Executes a pipeline with specified parameters and inputs, then evaluates its outputs using an evaluation pipeline.
    """

    @staticmethod
    def _override_pipeline(pipeline: Pipeline, parameter_overrides: Optional[Dict[str, Any]]) -> Pipeline:
        def component_pre_init_callback(name: str, cls: Type, init_params: Dict[str, Any]):  # pylint: disable=unused-argument
            assert parameter_overrides is not None
            overrides = parameter_overrides.get(name)
            if overrides:
                init_params.update(overrides)

        def validate_overrides():
            if parameter_overrides is None:
                return

            pipeline_components = pipeline.inputs(include_components_with_connected_inputs=True).keys()
            for component_name in parameter_overrides.keys():
                if component_name not in pipeline_components:
                    raise ValueError(f"Cannot override non-existent component '{component_name}'")

        callbacks = DeserializationCallbacks(component_pre_init_callback)
        if parameter_overrides:
            validate_overrides()
            serialized_pipeline = pipeline.dumps()
            pipeline = Pipeline.loads(serialized_pipeline, callbacks=callbacks)

        return pipeline

    @abstractmethod
    def run(
        self, inputs: EvalRunInputT, *, overrides: Optional[EvalRunOverridesT] = None, run_name: Optional[str] = None
    ) -> EvalRunOutputT:
        """
        Launch an evaluation run.

        :param inputs:
            Inputs to the evaluated and evaluation pipelines.
        :param overrides:
            Overrides for the harness.
        :param run_name:
            A name for the evaluation run.
        :returns:
            The output of the evaluation pipeline.
        """
