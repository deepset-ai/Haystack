# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
from typing import Tuple, Optional, List, Iterable, Dict

import logging
import itertools

import networkx

from canals.errors import PipelineConnectError, PipelineValidationError
from canals.pipeline.sockets import InputSocket, OutputSocket
from canals.component.input_output import ComponentInput


logger = logging.getLogger(__name__)


def parse_connection_name(connection: str) -> Tuple[str, Optional[str]]:
    """
    Returns component-connection pairs from a connect_to/from string
    """
    if "." in connection:
        split_str = connection.split(".", maxsplit=1)
        return (split_str[0], split_str[1])
    return connection, None


def find_pipeline_inputs(graph: networkx.MultiDiGraph) -> Dict[str, List[InputSocket]]:
    """
    Collect components that have disconnected input sockets. Note that this method returns *ALL* disconnected
    input sockets, including all such sockets with default values.
    """
    return {
        node: [socket for socket in data.get("input_sockets", {}).values() if not socket.taken_by]
        for node, data in graph.nodes(data=True)
    }


def find_pipeline_outputs(graph) -> Dict[str, List[OutputSocket]]:
    """
    Collect components that have disconnected output sockets. They define the pipeline output.
    """
    return {
        node: list(data.get("output_sockets", {}).values())
        for node, data in graph.nodes(data=True)
        if not graph.out_edges(node)
    }


def validate_pipeline_input(
    graph: networkx.MultiDiGraph, input_values: Dict[str, ComponentInput]
) -> Dict[str, ComponentInput]:
    """
    Make sure the pipeline is properly built and that the input received makes sense.
    Returns the input values, validated and updated at need.
    """
    input_components = find_pipeline_inputs(graph)
    if not any(sockets for sockets in find_pipeline_inputs(graph).values()):
        raise PipelineValidationError("This pipeline has no inputs.")

    # Make sure the input keys are all nodes of the pipeline
    unknown_components = [key for key in input_values.keys() if not key in graph.nodes]
    if unknown_components:
        raise ValueError(f"Pipeline received data for unknown component(s): {', '.join(unknown_components)}")

    # Make sure all necessary sockets are connected
    _validate_input_sockets_are_connected(graph, input_values)

    # Make sure that the pipeline input is only sent to nodes that won't receive data from other nodes
    _validate_nodes_receive_only_expected_input(graph, input_values)

    # Make sure variadic input components are receiving lists
    for component in input_components.keys():
        if graph.nodes[component]["variadic_input"] and component in input_values.keys():
            for key, value in input_values[component].__dict__.items():  # should be just one
                if not isinstance(value, Iterable):
                    setattr(input_values[component], key, [value])

    return input_values


def find_unambiguous_connection(
    from_node: str, to_node: str, from_sockets: List[OutputSocket], to_sockets: List[InputSocket]
) -> Tuple[OutputSocket, InputSocket]:
    """
    Find one single possible connection between two lists of sockets.
    """
    # List all combinations of sockets that match by type
    possible_connections = [
        (out_sock, in_sock)
        for out_sock, in_sock in itertools.product(from_sockets, to_sockets)
        if not in_sock.taken_by and out_sock.type == in_sock.type
    ]

    # No connections seem to be possible
    if not possible_connections:
        connections_status_str = _connections_status(
            from_node=from_node, from_sockets=from_sockets, to_node=to_node, to_sockets=to_sockets
        )
        raise PipelineConnectError(
            f"Cannot connect '{from_node}' with '{to_node}': "
            f"no matching connections available.\n{connections_status_str}"
        )

    # There's more than one possible connection
    if len(possible_connections) > 1:
        # Try to match by name
        name_matches = [
            (out_sock, in_sock) for out_sock, in_sock in possible_connections if in_sock.name == out_sock.name
        ]
        if len(name_matches) != 1:
            # TODO allow for multiple connections at once if there is no ambiguity?
            # TODO give priority to sockets that have no default values?
            connections_status_str = _connections_status(
                from_node=from_node, from_sockets=from_sockets, to_node=to_node, to_sockets=to_sockets
            )
            raise PipelineConnectError(
                f"Cannot connect '{from_node}' with '{to_node}': more than one connection is possible "
                "between these components. Please specify the connection name, like: "
                f"pipeline.connect('component_1.output_value', 'component_2.input_value').\n{connections_status_str}"
            )

    return possible_connections[0]


def _connections_status(from_node: str, to_node: str, from_sockets: List[OutputSocket], to_sockets: List[InputSocket]):
    """
    Lists the status of the sockets, for error messages.
    """
    from_sockets_list = "\n".join([f" - {socket.name} ({socket.type.__name__})" for socket in from_sockets])
    to_sockets_list = "\n".join(
        [
            f" - {socket.name} ({socket.type.__name__}, {'taken by '+socket.taken_by if socket.taken_by else 'available'})"
            for socket in to_sockets
        ]
    )
    return f"'{from_node}':\n{from_sockets_list}\n'{to_node}':\n{to_sockets_list}"


def _validate_input_sockets_are_connected(graph: networkx.MultiDiGraph, input_values: Dict[str, ComponentInput]):
    """
    Make sure all the inputs nodes are receiving all the values they need, either from the Pipeline's input or from
    other nodes.
    """
    valid_inputs = find_pipeline_inputs(graph)
    for node, sockets in valid_inputs.items():
        for socket in sockets:
            node_instance = graph.nodes[node]["instance"]
            input_in_node_defaults = hasattr(node_instance, "defaults") and socket.name in node_instance.defaults
            inputs_for_node = input_values.get(node)
            if not input_in_node_defaults and (not inputs_for_node or not socket.name in inputs_for_node.names()):
                raise ValueError(f"Missing input: {node}.{socket.name}")


def _validate_nodes_receive_only_expected_input(graph: networkx.MultiDiGraph, input_values: Dict[str, ComponentInput]):
    """
    Make sure that every input node is only receiving input values from EITHER the pipeline's input or another node,
    but never from both.
    """
    for node, input_data in input_values.items():
        for socket_name in input_data.names():
            if not getattr(input_data, socket_name):
                continue
            if not socket_name in graph.nodes[node]["input_sockets"].keys():
                raise ValueError(f"Component {node} is not expecting any input value called {socket_name}")

            taken_by = graph.nodes[node]["input_sockets"][socket_name].taken_by
            if taken_by:
                raise ValueError(f"The input {socket_name} of {node} is already taken by node {taken_by}")
