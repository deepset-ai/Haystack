# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from typing import List, Optional

from haystack.core.component import component


@component
class AddFixedValue:
    """
    Adds two values together.
    """

    def __init__(self, add: int = 1):
        self.add = add

    @component.output_types(result=int)
    def run(self, value: int, add: Optional[int] = None):
        """
        Adds two values together.
        """
        if add is None:
            add = self.add
        return {"result": value + add}


@component
class AddFixedValueBatch:
    """
    Adds two values together.
    """

    def __init__(self, add: int = 1):
        self.add = add

    @component.output_types(result=List[int])
    def run(self, value: List[int], add: Optional[List[int]] = None):
        """
        Adds two values together.
        """
        if add is None:
            add = [self.add] * len(value)
        return {"result": [v + a for v, a in zip(value, add)]}
