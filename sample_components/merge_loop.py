# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
from typing import List, Any

from canals import component


class MergeLoop:
    @staticmethod
    def create(expected_type: type, inputs: List[str]):
        """
        Takes multiple inputs and returns the first one that is not None.
        """

        @component
        class MergeLoopImpl:
            """
            Implementation of MergeLoop()
            """

            @component.return_types(value=expected_type)
            @component.run_method_types(**{input_name: expected_type for input_name in inputs})
            def run(self, **kwargs: Any):
                """
                :param kwargs: find the first non-None value and return it.
                """
                for v in kwargs.values():
                    if v is not None:
                        return {"value": v}
                return {"value": None}

        return MergeLoopImpl()

    def __init__(self):
        raise NotImplementedError("use MergeLoop.create()")
