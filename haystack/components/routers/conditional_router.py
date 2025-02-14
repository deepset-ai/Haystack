# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import ast
import contextlib
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Set, Union, get_args, get_origin
from warnings import warn

from jinja2 import Environment, TemplateSyntaxError, meta
from jinja2.nativetypes import NativeEnvironment, NativeTemplate, Template
from jinja2.sandbox import SandboxedEnvironment

from haystack import component, default_from_dict, default_to_dict, logging
from haystack.dataclasses import Answer, ByteStream, ChatMessage, Document, SparseEmbedding, StreamingChunk
from haystack.utils import deserialize_callable, deserialize_type, serialize_callable, serialize_type

logger = logging.getLogger(__name__)

haystack_dataclass_types = (ByteStream, ChatMessage, Document, Answer, SparseEmbedding, StreamingChunk)


class NoRouteSelectedException(Exception):
    """Exception raised when no route is selected in ConditionalRouter."""


class RouteConditionException(Exception):
    """Exception raised when there is an error parsing or evaluating the condition expression in ConditionalRouter."""


class NativeSandboxedTemplate(NativeTemplate, Template):
    """
    A template class that returns native Python objects and also respects the sandbox security checks.
    """

    pass


class NativeSandboxedEnvironment(SandboxedEnvironment, NativeEnvironment):
    """
    An environment that combines sandbox restrictions with native rendering.
    """

    # We tell the environment to use our custom template class by default.

    def from_string(self, source):
        """
        Override from_string to ensure the sandbox logic + native logic are used together.
        """
        template_class = NativeSandboxedTemplate

        return SandboxedEnvironment.from_string(self, source, template_class=template_class)

    def is_safe_attribute(self, obj, attr="", value=""):
        """
        Whitelist Haystack dataclasses so the sandbox won't block them.
        """

        if not isinstance(obj, haystack_dataclass_types):
            return False

        # Otherwise, fallback to the default sandbox behavior
        return SandboxedEnvironment.is_safe_attribute(self, obj, attr, value)


@component
class ConditionalRouter:
    """
    Routes data based on specific conditions.

    You define these conditions in a list of dictionaries called `routes`.
    Each dictionary in this list represents a single route. Each route has these four elements:
    - `condition`: A Jinja2 string expression that determines if the route is selected.
    - `output`: A Jinja2 expression defining the route's output value.
    - `output_type`: The type of the output data (for example, `str`, `List[int]`).
    - `output_name`: The name you want to use to publish `output`. This name is used to connect
    the router to other components in the pipeline.

    ### Usage example

    ```python
    from typing import List
    from haystack.components.routers import ConditionalRouter

    routes = [
        {
            "condition": "{{streams|length > 2}}",
            "output": "{{streams}}",
            "output_name": "enough_streams",
            "output_type": List[int],
        },
        {
            "condition": "{{streams|length <= 2}}",
            "output": "{{streams}}",
            "output_name": "insufficient_streams",
            "output_type": List[int],
        },
    ]
    router = ConditionalRouter(routes)
    # When 'streams' has more than 2 items, 'enough_streams' output will activate, emitting the list [1, 2, 3]
    kwargs = {"streams": [1, 2, 3], "query": "Haystack"}
    result = router.run(**kwargs)
    assert result == {"enough_streams": [1, 2, 3]}
    ```

    In this example, we configure two routes. The first route sends the 'streams' value to 'enough_streams' if the
    stream count exceeds two. The second route directs 'streams' to 'insufficient_streams' if there
    are two or fewer streams.

    In the pipeline setup, the Router connects to other components using the output names. For example,
    'enough_streams' might connect to a component that processes streams, while
    'insufficient_streams' might connect to a component that fetches more streams.


    Here is a pipeline that uses `ConditionalRouter` and routes the fetched `ByteStreams` to
    different components depending on the number of streams fetched:

    ```python
    from typing import List
    from haystack import Pipeline
    from haystack.dataclasses import ByteStream
    from haystack.components.routers import ConditionalRouter

    routes = [
        {
            "condition": "{{streams|length > 2}}",
            "output": "{{streams}}",
            "output_name": "enough_streams",
            "output_type": List[ByteStream],
        },
        {
            "condition": "{{streams|length <= 2}}",
            "output": "{{streams}}",
            "output_name": "insufficient_streams",
            "output_type": List[ByteStream],
        },
    ]

    pipe = Pipeline()
    pipe.add_component("router", router)
    ...
    pipe.connect("router.enough_streams", "some_component_a.streams")
    pipe.connect("router.insufficient_streams", "some_component_b.streams_or_some_other_input")
    ...
    ```
    """

    def __init__(  # pylint: disable=too-many-positional-arguments
        self,
        routes: List[Dict],
        custom_filters: Optional[Dict[str, Callable]] = None,
        unsafe: bool = False,
        validate_output_type: bool = False,
        optional_variables: Optional[List[str]] = None,
    ):
        """
        Initializes the `ConditionalRouter` with a list of routes detailing the conditions for routing.

        :param routes: A list of dictionaries, each defining a route.
            Each route has these four elements:
            - `condition`: A Jinja2 string expression that determines if the route is selected.
            - `output`: A Jinja2 expression defining the route's output value.
            - `output_type`: The type of the output data (for example, `str`, `List[int]`).
            - `output_name`: The name you want to use to publish `output`. This name is used to connect
            the router to other components in the pipeline.
        :param custom_filters: A dictionary of custom Jinja2 filters used in the condition expressions.
            For example, passing `{"my_filter": my_filter_fcn}` where:
            - `my_filter` is the name of the custom filter.
            - `my_filter_fcn` is a callable that takes `my_var:str` and returns `my_var[:3]`.
              `{{ my_var|my_filter }}` can then be used inside a route condition expression:
                `"condition": "{{ my_var|my_filter == 'foo' }}"`.
        :param unsafe:
            Enable execution of arbitrary code in the Jinja template.
            This should only be used if you trust the source of the template as it can be lead to remote code execution.
        :param validate_output_type:
            Enable validation of routes' output.
            If a route output doesn't match the declared type a ValueError is raised running.
        :param optional_variables:
            A list of variable names that are optional in your route conditions and outputs.
            If these variables are not provided at runtime, they will be set to `None`.
            This allows you to write routes that can handle missing inputs gracefully without raising errors.

            Example usage with a default fallback route in a Pipeline:
            ```python
            from haystack import Pipeline
            from haystack.components.routers import ConditionalRouter

            routes = [
                {
                    "condition": '{{ path == "rag" }}',
                    "output": "{{ question }}",
                    "output_name": "rag_route",
                    "output_type": str
                },
                {
                    "condition": "{{ True }}",  # fallback route
                    "output": "{{ question }}",
                    "output_name": "default_route",
                    "output_type": str
                }
            ]

            router = ConditionalRouter(routes, optional_variables=["path"])
            pipe = Pipeline()
            pipe.add_component("router", router)

            # When 'path' is provided in the pipeline:
            result = pipe.run(data={"router": {"question": "What?", "path": "rag"}})
            assert result["router"] == {"rag_route": "What?"}

            # When 'path' is not provided, fallback route is taken:
            result = pipe.run(data={"router": {"question": "What?"}})
            assert result["router"] == {"default_route": "What?"}
            ```

            This pattern is particularly useful when:
            - You want to provide default/fallback behavior when certain inputs are missing
            - Some variables are only needed for specific routing conditions
            - You're building flexible pipelines where not all inputs are guaranteed to be present
        """
        self.routes: List[dict] = routes
        self.custom_filters = custom_filters or {}
        self._unsafe = unsafe
        self._validate_output_type = validate_output_type
        self.optional_variables = optional_variables or []

        # Create a Jinja environment to inspect variables in the condition templates
        if self._unsafe:
            msg = (
                "Unsafe mode is enabled. This allows execution of arbitrary code in the Jinja template. "
                "Use this only if you trust the source of the template."
            )
            warn(msg)

        self._env = NativeEnvironment() if self._unsafe else SandboxedEnvironment()
        self._custom_env = NativeSandboxedEnvironment()
        self._env.filters.update(self.custom_filters)

        self._validate_routes(routes)
        # Inspect the routes to determine input and output types.
        input_types: Set[str] = set()  # let's just store the name, type will always be Any
        output_types: Dict[str, str] = {}

        for route in routes:
            # extract inputs
            route_input_names = self._extract_variables(self._env, [route["output"], route["condition"]])
            input_types.update(route_input_names)

            # extract outputs
            output_types.update({route["output_name"]: route["output_type"]})

        # remove optional variables from mandatory input types
        mandatory_input_types = input_types - set(self.optional_variables)

        # warn about unused optional variables
        unused_optional_vars = set(self.optional_variables) - input_types if self.optional_variables else None
        if unused_optional_vars:
            msg = (
                f"The following optional variables are specified but not used in any route: {unused_optional_vars}. "
                "Check if there's a typo in variable names."
            )
            # intentionally using both warn and logger
            warn(msg, UserWarning)
            logger.warning(msg)

        # add mandatory input types
        component.set_input_types(self, **{var: Any for var in mandatory_input_types})

        # now add optional input types
        for optional_var_name in self.optional_variables:
            component.set_input_type(self, name=optional_var_name, type=Any, default=None)

        # set output types
        component.set_output_types(self, **output_types)

    def to_dict(self) -> Dict[str, Any]:
        """
        Serializes the component to a dictionary.

        :returns:
            Dictionary with serialized data.
        """
        for route in self.routes:
            # output_type needs to be serialized to a string
            route["output_type"] = serialize_type(route["output_type"])
        se_filters = {name: serialize_callable(filter_func) for name, filter_func in self.custom_filters.items()}
        return default_to_dict(
            self,
            routes=self.routes,
            custom_filters=se_filters,
            unsafe=self._unsafe,
            validate_output_type=self._validate_output_type,
            optional_variables=self.optional_variables,
        )

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ConditionalRouter":
        """
        Deserializes the component from a dictionary.

        :param data:
            The dictionary to deserialize from.
        :returns:
            The deserialized component.
        """
        init_params = data.get("init_parameters", {})
        routes = init_params.get("routes")
        for route in routes:
            # output_type needs to be deserialized from a string to a type
            route["output_type"] = deserialize_type(route["output_type"])

        # Since the custom_filters are typed as optional in the init signature, we catch the
        # case where they are not present in the serialized data and set them to an empty dict.
        custom_filters = init_params.get("custom_filters", {})
        if custom_filters is not None:
            for name, filter_func in custom_filters.items():
                init_params["custom_filters"][name] = deserialize_callable(filter_func) if filter_func else None
        return default_from_dict(cls, data)

    def run(self, **kwargs):
        """
        Executes the routing logic.

        Executes the routing logic by evaluating the specified boolean condition expressions for each route in the
        order they are listed. The method directs the flow of data to the output specified in the first route whose
        `condition` is True.

        :param kwargs: All variables used in the `condition` expressed in the routes. When the component is used in a
            pipeline, these variables are passed from the previous component's output.

        :returns: A dictionary where the key is the `output_name` of the selected route and the value is the `output`
            of the selected route.

        :raises NoRouteSelectedException:
            If no `condition' in the routes is `True`.
        :raises RouteConditionException:
            If there is an error parsing or evaluating the `condition` expression in the routes.
        :raises ValueError:
            If type validation is enabled and route type doesn't match actual value type.
        """
        # Create a Jinja native environment to evaluate the condition templates as Python expressions
        for route in self.routes:
            try:
                t = self._env.from_string(route["condition"])
                rendered = t.render(**kwargs)
                if not self._unsafe:
                    rendered = ast.literal_eval(rendered)
                if not rendered:
                    continue
                # We now evaluate the `output` expression to determine the route output
                t_output = self._custom_env.from_string(route["output"])
                output = t_output.render(**kwargs)

                # Check if output is a list/sequence and validate accordingly
                if isinstance(output, (list, tuple)):
                    if all(self._custom_env.is_safe_attribute(item) for item in output):
                        pass
                elif self._custom_env.is_safe_attribute(output):
                    pass

                # We suppress the exception in case the output is already a string, otherwise
                # we try to evaluate it and would fail.
                # This must be done cause the output could be different literal structures.
                # This doesn't support any user types.
                else:
                    with contextlib.suppress(Exception):
                        if not self._unsafe and isinstance(output, str):
                            output = ast.literal_eval(output)
            except Exception as e:
                msg = f"Error evaluating condition for route '{route}': {e}"
                raise RouteConditionException(msg) from e

            if self._validate_output_type and not self._output_matches_type(output, route["output_type"]):
                msg = f"""Route '{route["output_name"]}' type doesn't match expected type"""
                raise ValueError(msg)

            # and return the output as a dictionary under the output_name key
            return {route["output_name"]: output}

        raise NoRouteSelectedException(f"No route fired. Routes: {self.routes}")

    def _validate_routes(self, routes: List[Dict]):
        """
        Validates a list of routes.

        :param routes: A list of routes.
        """
        for route in routes:
            try:
                keys = set(route.keys())
            except AttributeError:
                raise ValueError(f"Route must be a dictionary, got: {route}")

            mandatory_fields = {"condition", "output", "output_type", "output_name"}
            has_all_mandatory_fields = mandatory_fields.issubset(keys)
            if not has_all_mandatory_fields:
                raise ValueError(
                    f"Route must contain 'condition', 'output', 'output_type' and 'output_name' fields: {route}"
                )
            for field in ["condition", "output"]:
                if not self._validate_template(self._env, route[field]):
                    raise ValueError(f"Invalid template for field '{field}': {route[field]}")

    def _extract_variables(self, env: Environment, templates: List[str]) -> Set[str]:
        """
        Extracts all variables from a list of Jinja template strings.

        :param env: A Jinja environment.
        :param templates: A list of Jinja template strings.
        :returns: A set of variable names.
        """
        variables = set()
        for template in templates:
            ast = env.parse(template)
            variables.update(meta.find_undeclared_variables(ast))
        return variables

    def _validate_template(self, env: Environment, template_text: str):
        """
        Validates a template string by parsing it with Jinja.

        :param env: A Jinja environment.
        :param template_text: A Jinja template string.
        :returns: `True` if the template is valid, `False` otherwise.
        """
        try:
            env.parse(template_text)
            return True
        except TemplateSyntaxError:
            return False

    def _output_matches_type(self, value: Any, expected_type: type):  # noqa: PLR0911 # pylint: disable=too-many-return-statements
        """
        Checks whether `value` type matches the `expected_type`.
        """
        # Handle Any type
        if expected_type is Any:
            return True

        # Get the origin type (List, Dict, etc) and type arguments
        origin = get_origin(expected_type)
        args = get_args(expected_type)

        # Handle basic types (int, str, etc)
        if origin is None:
            return isinstance(value, expected_type)

        # Handle Sequence types (List, Tuple, etc)
        if isinstance(origin, type) and issubclass(origin, Sequence):
            if not isinstance(value, Sequence):
                return False
            # Empty sequence is valid
            if not value:
                return True
            # Check each element against the sequence's type parameter
            return all(self._output_matches_type(item, args[0]) for item in value)

        # Handle basic types (int, str, etc)
        if origin is None:
            return isinstance(value, expected_type)

        # Handle Mapping types (Dict, etc)
        if isinstance(origin, type) and issubclass(origin, Mapping):
            if not isinstance(value, Mapping):
                return False
            # Empty mapping is valid
            if not value:
                return True
            key_type, value_type = args
            # Check all keys and values match their respective types
            return all(
                self._output_matches_type(k, key_type) and self._output_matches_type(v, value_type)
                for k, v in value.items()
            )

        # Handle Union types (including Optional)
        if origin is Union:
            return any(self._output_matches_type(value, arg) for arg in args)

        return False
