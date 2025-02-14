# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from abc import ABC, abstractmethod
from typing import List, Optional

from pandas import DataFrame


class BaseEvaluationRunResult(ABC):
    """
    Represents the results of an evaluation run.
    """

    @abstractmethod
    def to_pandas(self) -> "DataFrame":
        """
        Creates a Pandas DataFrame containing the scores of each metric for every input sample.

        :returns:
            Pandas DataFrame with the scores.
        """

    @abstractmethod
    def score_report(self) -> "DataFrame":
        """
        Transforms the results into a Pandas DataFrame with the aggregated scores for each metric.

        :returns:
            Pandas DataFrame with the aggregated scores.
        """

    @abstractmethod
    def comparative_individual_scores_report(
        self, other: "BaseEvaluationRunResult", keep_columns: Optional[List[str]] = None
    ) -> "DataFrame":
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
