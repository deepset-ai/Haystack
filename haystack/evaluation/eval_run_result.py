# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from copy import deepcopy
from typing import Any, Dict, List, Optional
from warnings import warn

from pandas import DataFrame
from pandas import concat as pd_concat

from .base_eval_run_result import BaseEvaluationRunResult


class EvaluationRunResult(BaseEvaluationRunResult):
    """
    Contains the inputs and the outputs of an evaluation pipeline and provides methods to inspect them.
    """

    def __init__(self, run_name: str, inputs: Dict[str, List[Any]], results: Dict[str, Dict[str, Any]]):
        """
        Initialize a new evaluation run result.

        :param run_name:
            Name of the evaluation run.
        :param inputs:
            Dictionary containing the inputs used for the run.
            Each key is the name of the input and its value is
            a list of input values. The length of the lists should
            be the same.
        :param results:
            Dictionary containing the results of the evaluators
            used in the evaluation pipeline. Each key is the name
            of the metric and its value is dictionary with the following
            keys:
                - 'score': The aggregated score for the metric.
                - 'individual_scores': A list of scores for each input sample.
        """
        self.run_name = run_name
        self.inputs = deepcopy(inputs)
        self.results = deepcopy(results)

        if len(inputs) == 0:
            raise ValueError("No inputs provided.")
        if len({len(l) for l in inputs.values()}) != 1:
            raise ValueError("Lengths of the inputs should be the same.")

        expected_len = len(next(iter(inputs.values())))

        for metric, outputs in results.items():
            if "score" not in outputs:
                raise ValueError(f"Aggregate score missing for {metric}.")
            if "individual_scores" not in outputs:
                raise ValueError(f"Individual scores missing for {metric}.")

            if len(outputs["individual_scores"]) != expected_len:
                raise ValueError(
                    f"Length of individual scores for '{metric}' should be the same as the inputs. "
                    f"Got {len(outputs['individual_scores'])} but expected {expected_len}."
                )

    def score_report(self) -> DataFrame:
        """
        Transforms the results into a Pandas DataFrame with the aggregated scores for each metric.

        :returns:
            Pandas DataFrame with the aggregated scores.
        """
        results = {k: v["score"] for k, v in self.results.items()}
        df = DataFrame.from_dict(results, orient="index", columns=["score"]).reset_index()
        df.columns = ["metrics", "score"]
        return df

    def to_pandas(self) -> DataFrame:
        """
        Creates a Pandas DataFrame containing the scores of each metric for every input sample.

        :returns:
            Pandas DataFrame with the scores.
        """
        inputs_columns = list(self.inputs.keys())
        inputs_values = list(self.inputs.values())
        inputs_values = list(map(list, zip(*inputs_values)))  # transpose the values
        df_inputs = DataFrame(inputs_values, columns=inputs_columns)

        scores_columns = list(self.results.keys())
        scores_values = [v["individual_scores"] for v in self.results.values()]
        scores_values = list(map(list, zip(*scores_values)))  # transpose the values
        df_scores = DataFrame(scores_values, columns=scores_columns)

        return df_inputs.join(df_scores)

    def comparative_individual_scores_report(
        self, other: "BaseEvaluationRunResult", keep_columns: Optional[List[str]] = None
    ) -> DataFrame:
        """
        Creates a Pandas DataFrame with the scores for each metric in the results of two different evaluation runs.

        The inputs to both evaluation runs is assumed to be the same.

        :param other:
            Results of another evaluation run to compare with.
        :param keep_columns:
            List of common column names to keep from the inputs of the evaluation runs to compare.
        :returns:
            Pandas DataFrame with the score comparison.
        """
        if not isinstance(other, EvaluationRunResult):
            raise ValueError("Comparative scores can only be computed between EvaluationRunResults.")

        this_name = self.run_name
        other_name = other.run_name
        if this_name == other_name:
            warn(f"The run names of the two evaluation results are the same ('{this_name}')")
            this_name = f"{this_name}_first"
            other_name = f"{other_name}_second"

        if self.inputs.keys() != other.inputs.keys():
            warn(f"The input columns differ between the results; using the input columns of '{this_name}'.")

        pipe_a_df = self.to_pandas()
        pipe_b_df = other.to_pandas()

        if keep_columns is None:
            ignore = list(self.inputs.keys())
        else:
            ignore = [col for col in list(self.inputs.keys()) if col not in keep_columns]

        pipe_b_df.drop(columns=ignore, inplace=True, errors="ignore")
        pipe_b_df.columns = [f"{other_name}_{column}" for column in pipe_b_df.columns]  # type: ignore
        pipe_a_df.columns = [f"{this_name}_{col}" if col not in ignore else col for col in pipe_a_df.columns]  # type: ignore

        results_df = pd_concat([pipe_a_df, pipe_b_df], axis=1)

        return results_df
