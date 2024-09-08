# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any, List, Optional, Union

import arrow
from jinja2 import Environment, nodes
from jinja2.ext import Extension


class Jinja2TimeExtension(Extension):
    # Syntax for current date
    tags = {"now"}

    def __init__(self, environment: Environment):  # pylint: disable=useless-parent-delegation
        """
        Initializes the JinjaTimeExtension object.

        :param environment: The Jinja2 environment to initialize the extension with.
            It provides the context where the extension will operate.
        """
        super().__init__(environment)

    @staticmethod
    def _get_datetime(
        timezone: str,
        operator: Optional[str] = None,
        offset: Optional[str] = None,
        datetime_format: Optional[str] = None,
    ) -> str:
        """
        Get the current datetime based on timezone, apply any offset if provided, and format the result.

        :param timezone: The timezone string (e.g., 'UTC' or 'America/New_York') for which the current
            time should be fetched.
        :param operator: The operator ('+' or '-') to apply to the offset (used for adding/subtracting intervals).
            Defaults to None if no offset is applied.
        :param offset: The offset string in the format 'interval=value' (e.g., 'hours=2,days=1') specifying how much
            to adjust the datetime. The intervals can be any valid interval accepted
            by Arrow (e.g., hours, days, weeks, months). Defaults to None if no adjustment is needed.
        :param datetime_format: The format string to use for formatting the output datetime.
            Defaults to '%Y-%m-%d %H:%M:%S' if not provided.
        """
        try:
            dt = arrow.now(timezone)
        except Exception as e:
            raise ValueError(f"Invalid timezone {timezone}: {e}")

        if offset is not None and operator is not None:
            try:
                # Parse the offset and apply it to the datetime object
                replace_params = {
                    interval.strip(): float(operator + value.strip())
                    for param in offset.split(",")
                    for interval, value in [param.split("=")]
                }
                # Shift the datetime fields based on the parsed offset
                dt = dt.shift(**replace_params)
            except (ValueError, AttributeError) as e:
                raise ValueError(f"Invalid offset or operator {offset}, {operator}: {e}")

        # Use the provided format or fallback to the default one
        datetime_format = datetime_format or "%Y-%m-%d %H:%M:%S"

        return dt.strftime(datetime_format)

    def parse(self, parser: Any) -> Union[nodes.Node, List[nodes.Node]]:
        """
        Parse the template expression to determine how to handle the datetime formatting.

        :param parser: The parser object that processes the template expressions and manages the syntax tree.
            It's used to interpret the template's structure.
        """
        lineno = next(parser.stream).lineno

        node = parser.parse_expression()

        # Check if a custom datetime format is provided after a comma
        datetime_format = parser.parse_expression() if parser.stream.skip_if("comma") else nodes.Const(None)

        if isinstance(node, (nodes.Add, nodes.Sub)):
            operator = "+" if isinstance(node, nodes.Add) else "-"
            # Call the _get_datetime method with the appropriate operator and offset
            call_method = self.call_method(
                "_get_datetime", [node.left, nodes.Const(operator), node.right, datetime_format], lineno=lineno
            )
        else:
            # If no Add/Subtract, just call _get_datetime for the current time without offsets
            call_method = self.call_method(
                "_get_datetime", [node, nodes.Const(None), nodes.Const(None), datetime_format], lineno=lineno
            )

        return nodes.Output([call_method], lineno=lineno)
