# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import pytest

from haystack.core.pipeline.component_checks import *
from haystack.core.pipeline.component_checks import _NO_OUTPUT_PRODUCED
from haystack.core.component.types import InputSocket, OutputSocket, Variadic, GreedyVariadic


import pandas as pd


@pytest.fixture
def basic_component():
    """Basic component with one mandatory and one optional input."""
    return {
        "instance": "mock_instance",
        "visits": 0,
        "input_sockets": {
            "mandatory_input": InputSocket("mandatory_input", int, senders=["previous_component"]),
            "optional_input": InputSocket("optional_input", str, default_value="default"),
        },
        "output_sockets": {"output": OutputSocket("output", int)},
    }


@pytest.fixture
def variadic_component():
    """Component with variadic input."""
    return {
        "instance": "mock_instance",
        "visits": 0,
        "input_sockets": {
            "variadic_input": InputSocket("variadic_input", Variadic[int], senders=["previous_component"]),
            "normal_input": InputSocket("normal_input", str, senders=["another_component"]),
        },
        "output_sockets": {"output": OutputSocket("output", int)},
    }


@pytest.fixture
def greedy_variadic_component():
    """Component with greedy variadic input."""
    return {
        "instance": "mock_instance",
        "visits": 0,
        "input_sockets": {
            "greedy_input": InputSocket(
                "greedy_input", GreedyVariadic[int], senders=["previous_component", "other_component"]
            ),
            "normal_input": InputSocket("normal_input", str),
        },
        "output_sockets": {"output": OutputSocket("output", int)},
    }


@pytest.fixture
def input_socket_with_sender():
    """Regular input socket with a single sender."""
    socket = InputSocket("test_input", int)
    socket.senders = ["component1"]
    return socket


@pytest.fixture
def variadic_socket_with_senders():
    """Variadic input socket with multiple senders."""
    socket = InputSocket("test_variadic", Variadic[int])
    socket.senders = ["component1", "component2"]
    return socket


@pytest.fixture
def component_with_multiple_sockets(input_socket_with_sender, variadic_socket_with_senders):
    """Component with multiple input sockets including both regular and variadic."""
    return {
        "instance": "mock_instance",
        "input_sockets": {
            "socket1": input_socket_with_sender,
            "socket2": variadic_socket_with_senders,
            "socket3": InputSocket("socket3", str),  # No senders
        },
    }


@pytest.fixture
def regular_socket():
    """Regular input socket with one sender."""
    socket = InputSocket("regular", int)
    socket.senders = ["component1"]
    return socket


@pytest.fixture
def lazy_variadic_socket():
    """Lazy variadic input socket with multiple senders."""
    socket = InputSocket("lazy_variadic", Variadic[int])
    socket.senders = ["component1", "component2"]
    return socket


@pytest.fixture
def greedy_variadic_socket():
    """Greedy variadic input socket with multiple senders."""
    socket = InputSocket("greedy_variadic", GreedyVariadic[int])
    socket.senders = ["component1", "component2", "component3"]
    return socket


@pytest.fixture
def complex_component(regular_socket, lazy_variadic_socket, greedy_variadic_socket):
    """Component with all types of sockets."""
    return {
        "instance": "mock_instance",
        "input_sockets": {
            "regular": regular_socket,
            "lazy_var": lazy_variadic_socket,
            "greedy_var": greedy_variadic_socket,
        },
    }


class TestCanComponentRun:
    def test_component_with_all_mandatory_inputs_and_trigger(self, basic_component):
        """Checks that the component runs if all mandatory inputs are received and triggered."""
        inputs = {"mandatory_input": [{"sender": "previous_component", "value": 42}]}
        assert can_component_run(basic_component, inputs) is True

    def test_component_missing_mandatory_input(self, basic_component):
        """Checks that the component won't run if mandatory inputs are missing."""
        inputs = {"optional_input": [{"sender": "previous_component", "value": "test"}]}
        assert can_component_run(basic_component, inputs) is False

    # We added these tests because a component that returned a pandas dataframe caused the pipeline to fail.
    # Previously, we compared the value of the socket using '!=' which leads to an error with dataframes.
    # Instead, we use 'is not' to compare with the sentinel value.
    def test_sockets_with_ambiguous_truth_value(self, basic_component, greedy_variadic_socket, regular_socket):
        inputs = {
            "mandatory_input": [{"sender": "previous_component", "value": pd.DataFrame.from_dict([{"value": 42}])}]
        }

        assert are_all_sockets_ready(basic_component, inputs, only_check_mandatory=True) is True
        assert any_socket_value_from_predecessor_received(inputs["mandatory_input"]) is True
        assert any_socket_input_received(inputs["mandatory_input"]) is True
        assert (
            has_lazy_variadic_socket_received_all_inputs(
                basic_component["input_sockets"]["mandatory_input"], inputs["mandatory_input"]
            )
            is True
        )
        assert has_socket_received_all_inputs(greedy_variadic_socket, inputs["mandatory_input"]) is True
        assert has_socket_received_all_inputs(regular_socket, inputs["mandatory_input"]) is True

    def test_component_with_no_trigger_but_all_inputs(self, basic_component):
        """
        Test case where all mandatory inputs are present with valid values,
        but there is no trigger (no new input from predecessor, not first visit).
        """
        basic_component["visits"] = 1
        inputs = {"mandatory_input": [{"sender": None, "value": 42}]}
        assert can_component_run(basic_component, inputs) is False

    def test_component_with_multiple_visits(self, basic_component):
        """Checks that a component can still be triggered on subsequent visits by a predecessor."""
        basic_component["visits"] = 2
        inputs = {"mandatory_input": [{"sender": "previous_component", "value": 42}]}
        assert can_component_run(basic_component, inputs) is True

    def test_component_with_no_inputs_first_visit(self, basic_component):
        """Checks that a component with no input sockets can be triggered on its first visit."""
        basic_component["input_sockets"] = {}
        inputs = {}
        assert can_component_run(basic_component, inputs) is True

    def test_component_triggered_on_second_visit_with_new_input(self, basic_component):
        """
        Tests that a second visit is triggered if new predecessor input arrives
        (i.e. visits > 0, but a valid new input from a predecessor is provided).
        """
        # First, simulate that the component has already run once.
        basic_component["visits"] = 1

        # Now a predecessor provides a new input; this should re-trigger execution.
        inputs = {"mandatory_input": [{"sender": "previous_component", "value": 99}]}
        assert can_component_run(basic_component, inputs) is True


class TestHasAnyTrigger:
    def test_trigger_from_predecessor(self, basic_component):
        """Ensures that new data from a predecessor can trigger a component."""
        inputs = {"mandatory_input": [{"sender": "previous_component", "value": 42}]}
        assert has_any_trigger(basic_component, inputs) is True

    def test_trigger_from_user_first_visit(self, basic_component):
        """Checks that user input (sender=None) triggers the component on the first visit."""
        inputs = {"mandatory_input": [{"sender": None, "value": 42}]}
        assert has_any_trigger(basic_component, inputs) is True

    def test_no_trigger_from_user_after_first_visit(self, basic_component):
        """Checks that user input no longer triggers the component after the first visit."""
        basic_component["visits"] = 1
        inputs = {"mandatory_input": [{"sender": None, "value": 42}]}
        assert has_any_trigger(basic_component, inputs) is False

    def test_trigger_without_inputs_first_visit(self, basic_component):
        """Checks that a component with no inputs is triggered on the first visit."""
        basic_component["input_sockets"] = {}
        inputs = {}
        assert has_any_trigger(basic_component, inputs) is True

    def test_no_trigger_without_inputs_after_first_visit(self, basic_component):
        """Checks that on subsequent visits, no inputs means no trigger."""
        basic_component["input_sockets"] = {}
        basic_component["visits"] = 1
        inputs = {}
        assert has_any_trigger(basic_component, inputs) is False


class TestAllMandatorySocketsReady:
    def test_all_mandatory_sockets_filled(self, basic_component):
        """Checks that all mandatory sockets are ready when they have valid input."""
        inputs = {"mandatory_input": [{"sender": "previous_component", "value": 42}]}
        assert are_all_sockets_ready(basic_component, inputs) is True

    def test_missing_mandatory_socket(self, basic_component):
        """Ensures that if a mandatory socket is missing, the component is not ready."""
        inputs = {"optional_input": [{"sender": "previous_component", "value": "test"}]}
        assert are_all_sockets_ready(basic_component, inputs) is False

    def test_variadic_socket_with_input(self, variadic_component):
        """Verifies that a variadic socket is considered filled if it has at least one input."""
        inputs = {
            "variadic_input": [{"sender": "previous_component", "value": 42}],
            "normal_input": [{"sender": "previous_component", "value": "test"}],
        }
        assert are_all_sockets_ready(variadic_component, inputs) is True

    def test_greedy_variadic_socket(self, greedy_variadic_component):
        """Greedy variadic sockets are ready with at least one valid input."""
        inputs = {
            "greedy_input": [{"sender": "previous_component", "value": 42}],
            "normal_input": [{"sender": "previous_component", "value": "test"}],
        }
        assert are_all_sockets_ready(greedy_variadic_component, inputs) is True

    def test_greedy_variadic_socket_and_missing_mandatory(self, greedy_variadic_component):
        """All mandatory sockets need to be filled even with GreedyVariadic sockets."""
        inputs = {"greedy_input": [{"sender": "previous_component", "value": 42}]}
        assert are_all_sockets_ready(greedy_variadic_component, inputs, only_check_mandatory=True) is False

    def test_variadic_socket_no_input(self, variadic_component):
        """A variadic socket is not filled if it has zero valid inputs."""
        inputs = {"normal_input": [{"sender": "previous_component", "value": "test"}]}
        assert are_all_sockets_ready(variadic_component, inputs) is False

    def test_mandatory_and_optional_sockets(self):
        input_sockets = {
            "mandatory": InputSocket("mandatory", str, senders=["previous_component"]),
            "optional": InputSocket("optional", str, senders=["previous_component"], default_value="test"),
        }

        component = {"input_sockets": input_sockets}
        inputs = {"mandatory": [{"sender": "previous_component", "value": "hello"}]}
        assert are_all_sockets_ready(component, inputs) is False
        assert are_all_sockets_ready(component, inputs, only_check_mandatory=True) is True

    def test_empty_inputs(self, basic_component):
        """Checks that if there are no inputs at all, mandatory sockets are not ready."""
        inputs = {}
        assert are_all_sockets_ready(basic_component, inputs) is False

    def test_no_mandatory_sockets(self, basic_component):
        """Ensures that if there are no mandatory sockets, the component is considered ready."""
        basic_component["input_sockets"] = {
            "optional_1": InputSocket("optional_1", str, default_value="default1"),
            "optional_2": InputSocket("optional_2", str, default_value="default2"),
        }
        inputs = {}
        assert are_all_sockets_ready(basic_component, inputs) is True

    def test_multiple_mandatory_sockets(self, basic_component):
        """Checks readiness when multiple mandatory sockets are defined."""
        basic_component["input_sockets"] = {
            "mandatory_1": InputSocket("mandatory_1", int, senders=["previous_component"]),
            "mandatory_2": InputSocket("mandatory_2", str, senders=["some other component"]),
            "optional": InputSocket("optional", bool, default_value=False),
        }
        inputs = {
            "mandatory_1": [{"sender": "comp1", "value": 42}],
            "mandatory_2": [{"sender": "comp2", "value": "test"}],
        }
        assert are_all_sockets_ready(basic_component, inputs) is True

        # Missing one mandatory input
        inputs = {"mandatory_1": [{"sender": "comp1", "value": 42}], "optional": [{"sender": "comp3", "value": True}]}
        assert are_all_sockets_ready(basic_component, inputs) is False


class TestPredecessorInputDetection:
    def test_any_predecessors_provided_input_with_predecessor(self, component_with_multiple_sockets):
        """
        Tests detection of predecessor input when a valid predecessor sends data.
        """
        inputs = {"socket1": [{"sender": "component1", "value": 42}], "socket2": [{"sender": None, "value": "test"}]}
        assert any_predecessors_provided_input(component_with_multiple_sockets, inputs) is True

    def test_any_predecessors_provided_input_no_predecessor(self, component_with_multiple_sockets):
        """
        Checks that no predecessor inputs are detected if all senders are None (user inputs).
        """
        inputs = {"socket1": [{"sender": None, "value": 42}], "socket2": [{"sender": None, "value": "test"}]}
        assert any_predecessors_provided_input(component_with_multiple_sockets, inputs) is False

    def test_any_predecessors_provided_input_with_no_output(self, component_with_multiple_sockets):
        """
        Ensures that _NO_OUTPUT_PRODUCED from a predecessor is ignored in the predecessor detection.
        """
        inputs = {
            "socket1": [{"sender": "component1", "value": _NO_OUTPUT_PRODUCED}],
            "socket2": [{"sender": None, "value": "test"}],
        }
        assert any_predecessors_provided_input(component_with_multiple_sockets, inputs) is False

    def test_any_predecessors_provided_input_empty_inputs(self, component_with_multiple_sockets):
        """Ensures that empty inputs dictionary returns False."""
        inputs = {}
        assert any_predecessors_provided_input(component_with_multiple_sockets, inputs) is False


class TestSocketValueFromPredecessor:
    """
    Tests for `any_socket_value_from_predecessor_received`, verifying whether
    any predecessor component provided valid output to a socket.
    """

    @pytest.mark.parametrize(
        "socket_inputs, expected_result",
        [
            pytest.param([{"sender": "component1", "value": 42}], True, id="valid_input"),
            pytest.param([{"sender": "component1", "value": _NO_OUTPUT_PRODUCED}], False, id="no_output"),
            pytest.param([{"sender": None, "value": 42}], False, id="user_input"),
            pytest.param(
                [
                    {"sender": None, "value": 42},
                    {"sender": "component1", "value": _NO_OUTPUT_PRODUCED},
                    {"sender": "component2", "value": 100},
                ],
                True,
                id="mixed_inputs",
            ),
            pytest.param([], False, id="empty_list"),
        ],
    )
    def test_any_socket_value_from_predecessor_received(self, socket_inputs, expected_result):
        """
        Parametrized test to check whether any valid predecessor input
        exists in a list of socket inputs.
        """
        assert any_socket_value_from_predecessor_received(socket_inputs) == expected_result


class TestUserInputDetection:
    def test_has_user_input_with_user_input(self):
        """Checks that having a sender=None input means user input is present."""
        inputs = {"socket1": [{"sender": None, "value": 42}], "socket2": [{"sender": "component1", "value": "test"}]}
        assert has_user_input(inputs) is True

    def test_has_user_input_without_user_input(self):
        """Ensures that if all senders are component-based, there's no user input."""
        inputs = {
            "socket1": [{"sender": "component1", "value": 42}],
            "socket2": [{"sender": "component2", "value": "test"}],
        }
        assert has_user_input(inputs) is False

    def test_has_user_input_empty_inputs(self):
        """Checks that an empty inputs dict has no user input."""
        inputs = {}
        assert has_user_input(inputs) is False

    def test_has_user_input_with_no_output(self):
        """
        Even if the input value is _NO_OUTPUT_PRODUCED, if sender=None
        it still counts as user input being provided.
        """
        inputs = {"socket1": [{"sender": None, "value": _NO_OUTPUT_PRODUCED}]}
        assert has_user_input(inputs) is True


class TestPipelineInputCapability:
    def test_cannot_receive_inputs_no_senders(self):
        """Checks that a component with zero senders for each socket cannot receive pipeline inputs."""
        component = {"input_sockets": {"socket1": InputSocket("socket1", int), "socket2": InputSocket("socket2", str)}}
        assert can_not_receive_inputs_from_pipeline(component) is True

    def test_cannot_receive_inputs_with_senders(self, component_with_multiple_sockets):
        """If at least one socket has a sender, the component can receive pipeline inputs."""
        assert can_not_receive_inputs_from_pipeline(component_with_multiple_sockets) is False

    def test_cannot_receive_inputs_mixed_senders(self, input_socket_with_sender):
        """A single socket with a sender means the component can receive pipeline inputs."""
        component = {
            "input_sockets": {
                "socket1": input_socket_with_sender,
                "socket2": InputSocket("socket2", str),  # No senders
            }
        }
        assert can_not_receive_inputs_from_pipeline(component) is False


class TestSocketExecutionStatus:
    def test_regular_socket_predecessor_executed(self, input_socket_with_sender):
        """Verifies that if the correct sender provides a value, the socket is marked as executed."""
        socket_inputs = [{"sender": "component1", "value": 42}]
        assert all_socket_predecessors_executed(input_socket_with_sender, socket_inputs) is True

    def test_regular_socket_predecessor_not_executed(self, input_socket_with_sender):
        """If there are no inputs, the predecessor is not considered executed."""
        socket_inputs = []
        assert all_socket_predecessors_executed(input_socket_with_sender, socket_inputs) is False

    def test_regular_socket_with_wrong_predecessor(self, input_socket_with_sender):
        """Checks that a mismatch in sender means the socket is not yet executed."""
        socket_inputs = [{"sender": "component2", "value": 42}]
        assert all_socket_predecessors_executed(input_socket_with_sender, socket_inputs) is False

    def test_variadic_socket_all_predecessors_executed(self, variadic_socket_with_senders):
        """Variadic socket is executed only if all senders have produced at least one valid result."""
        socket_inputs = [{"sender": "component1", "value": 42}, {"sender": "component2", "value": 43}]
        assert all_socket_predecessors_executed(variadic_socket_with_senders, socket_inputs) is True

    def test_variadic_socket_partial_execution(self, variadic_socket_with_senders):
        """If only one of multiple senders produced an output, not all predecessors are executed."""
        socket_inputs = [{"sender": "component1", "value": 42}]
        assert all_socket_predecessors_executed(variadic_socket_with_senders, socket_inputs) is False

    def test_variadic_socket_with_user_input(self, variadic_socket_with_senders):
        """
        User input (sender=None) doesn't block the socket from being 'executed' if
        all named predecessors have also produced outputs.
        """
        socket_inputs = [
            {"sender": "component1", "value": 42},
            {"sender": None, "value": 43},
            {"sender": "component2", "value": 44},
        ]
        assert all_socket_predecessors_executed(variadic_socket_with_senders, socket_inputs) is True

    def test_variadic_socket_no_execution(self, variadic_socket_with_senders):
        """Empty inputs means no predecessor has executed."""
        socket_inputs = []
        assert all_socket_predecessors_executed(variadic_socket_with_senders, socket_inputs) is False


class TestSocketInputReceived:
    def test_any_socket_input_received_with_value(self):
        """Checks that if there's a non-_NO_OUTPUT_PRODUCED value, the socket is marked as having input."""
        socket_inputs = [{"sender": "component1", "value": 42}]
        assert any_socket_input_received(socket_inputs) is True

    def test_any_socket_input_received_with_no_output(self):
        """If all inputs are _NO_OUTPUT_PRODUCED, the socket has no effective input."""
        socket_inputs = [{"sender": "component1", "value": _NO_OUTPUT_PRODUCED}]
        assert any_socket_input_received(socket_inputs) is False

    def test_any_socket_input_received_mixed_inputs(self):
        """A single valid input among many is enough to consider the socket as having input."""
        socket_inputs = [{"sender": "component1", "value": _NO_OUTPUT_PRODUCED}, {"sender": "component2", "value": 42}]
        assert any_socket_input_received(socket_inputs) is True

    def test_any_socket_input_received_empty_list(self):
        """Empty list: no input received."""
        assert any_socket_input_received([]) is False


class TestLazyVariadicSocket:
    def test_lazy_variadic_all_inputs_received(self, variadic_socket_with_senders):
        """Lazy variadic socket is ready only if all named senders provided outputs."""
        socket_inputs = [{"sender": "component1", "value": 42}, {"sender": "component2", "value": 43}]
        assert has_lazy_variadic_socket_received_all_inputs(variadic_socket_with_senders, socket_inputs) is True

    def test_lazy_variadic_partial_inputs(self, variadic_socket_with_senders):
        """Partial inputs from only some senders is insufficient for a lazy variadic socket."""
        socket_inputs = [{"sender": "component1", "value": 42}]
        assert has_lazy_variadic_socket_received_all_inputs(variadic_socket_with_senders, socket_inputs) is False

    def test_lazy_variadic_with_no_output(self, variadic_socket_with_senders):
        """_NO_OUTPUT_PRODUCED from a sender doesn't count as valid input, so it's not fully received."""
        socket_inputs = [{"sender": "component1", "value": _NO_OUTPUT_PRODUCED}, {"sender": "component2", "value": 42}]
        assert has_lazy_variadic_socket_received_all_inputs(variadic_socket_with_senders, socket_inputs) is False

    def test_lazy_variadic_with_user_input(self, variadic_socket_with_senders):
        """
        User input doesn't block a lazy variadic socket, as long as all named senders
        also provided outputs.
        """
        socket_inputs = [
            {"sender": "component1", "value": 42},
            {"sender": None, "value": 43},
            {"sender": "component2", "value": 44},
        ]
        assert has_lazy_variadic_socket_received_all_inputs(variadic_socket_with_senders, socket_inputs) is True

    def test_lazy_variadic_empty_inputs(self, variadic_socket_with_senders):
        """No inputs at all means the lazy variadic socket hasn't received everything yet."""
        assert has_lazy_variadic_socket_received_all_inputs(variadic_socket_with_senders, []) is False


class TestSocketTypeDetection:
    def test_is_socket_lazy_variadic_with_lazy_socket(self, lazy_variadic_socket):
        """Ensures that a non-greedy variadic socket is detected as lazy."""
        assert is_socket_lazy_variadic(lazy_variadic_socket) is True

    def test_is_socket_lazy_variadic_with_greedy_socket(self, greedy_variadic_socket):
        """Greedy variadic sockets should not be marked as lazy."""
        assert is_socket_lazy_variadic(greedy_variadic_socket) is False

    def test_is_socket_lazy_variadic_with_regular_socket(self, regular_socket):
        """Regular sockets are not variadic at all."""
        assert is_socket_lazy_variadic(regular_socket) is False


class TestSocketInputCompletion:
    def test_regular_socket_complete(self, regular_socket):
        """A single valid input marks a regular socket as complete."""
        inputs = [{"sender": "component1", "value": 42}]
        assert has_socket_received_all_inputs(regular_socket, inputs) is True

    def test_regular_socket_incomplete(self, regular_socket):
        """_NO_OUTPUT_PRODUCED means the socket is not complete."""
        inputs = [{"sender": "component1", "value": _NO_OUTPUT_PRODUCED}]
        assert has_socket_received_all_inputs(regular_socket, inputs) is False

    def test_regular_socket_no_inputs(self, regular_socket):
        """No inputs at all means the socket is incomplete."""
        inputs = []
        assert has_socket_received_all_inputs(regular_socket, inputs) is False

    def test_lazy_variadic_socket_all_inputs(self, lazy_variadic_socket):
        """Lazy variadic socket is complete only if all senders have produced valid outputs."""
        inputs = [{"sender": "component1", "value": 42}, {"sender": "component2", "value": 43}]
        assert has_socket_received_all_inputs(lazy_variadic_socket, inputs) is True

    def test_lazy_variadic_socket_partial_inputs(self, lazy_variadic_socket):
        """Partial coverage of senders is insufficient for lazy variadic sockets."""
        inputs = [{"sender": "component1", "value": 42}]
        assert has_socket_received_all_inputs(lazy_variadic_socket, inputs) is False

    def test_lazy_variadic_socket_with_no_output(self, lazy_variadic_socket):
        """A sender that produces _NO_OUTPUT_PRODUCED does not fulfill the lazy socket requirement."""
        inputs = [{"sender": "component1", "value": 42}, {"sender": "component2", "value": _NO_OUTPUT_PRODUCED}]
        assert has_socket_received_all_inputs(lazy_variadic_socket, inputs) is False

    def test_greedy_variadic_socket_one_input(self, greedy_variadic_socket):
        """A greedy variadic socket is complete if it has at least one valid input."""
        inputs = [{"sender": "component1", "value": 42}]
        assert has_socket_received_all_inputs(greedy_variadic_socket, inputs) is True

    def test_greedy_variadic_socket_multiple_inputs(self, greedy_variadic_socket):
        """A greedy variadic socket with multiple inputs remains complete as soon as one is valid."""
        inputs = [{"sender": "component1", "value": 42}, {"sender": "component2", "value": 43}]
        assert has_socket_received_all_inputs(greedy_variadic_socket, inputs) is True

    def test_greedy_variadic_socket_no_valid_inputs(self, greedy_variadic_socket):
        """All _NO_OUTPUT_PRODUCED means the greedy socket is not complete."""
        inputs = [{"sender": "component1", "value": _NO_OUTPUT_PRODUCED}]
        assert has_socket_received_all_inputs(greedy_variadic_socket, inputs) is False


class TestPredecessorExecution:
    def test_all_predecessors_executed_complete(self, complex_component):
        """
        Checks that if all named senders produce valid outputs for each socket,
        then all predecessors are considered executed.
        """
        inputs = {
            "regular": [{"sender": "component1", "value": 42}],
            "lazy_var": [{"sender": "component1", "value": 42}, {"sender": "component2", "value": 43}],
            "greedy_var": [
                {"sender": "component1", "value": 42},
                {"sender": "component2", "value": 43},
                {"sender": "component3", "value": 44},
            ],
        }
        assert all_predecessors_executed(complex_component, inputs) is True

    def test_all_predecessors_executed_partial(self, complex_component):
        """If a lazy socket is missing one predecessor, not all predecessors are executed."""
        inputs = {
            "regular": [{"sender": "component1", "value": 42}],
            "lazy_var": [{"sender": "component1", "value": 42}],  # Missing component2
            "greedy_var": [{"sender": "component1", "value": 42}, {"sender": "component2", "value": 43}],
        }
        assert all_predecessors_executed(complex_component, inputs) is False

    def test_all_predecessors_executed_with_user_input(self, complex_component):
        """
        User input shouldn't affect predecessor execution for the lazy socket:
        we still need all named senders to produce output.
        """
        inputs = {
            "regular": [{"sender": "component1", "value": 42}],
            "lazy_var": [{"sender": "component1", "value": 42}, {"sender": None, "value": 43}],
            "greedy_var": [
                {"sender": "component1", "value": 42},
                {"sender": "component2", "value": 43},
                {"sender": "component3", "value": 44},
            ],
        }
        assert all_predecessors_executed(complex_component, inputs) is False


class TestLazyVariadicResolution:
    def test_lazy_variadic_sockets_all_resolved(self, complex_component):
        """Checks that lazy variadic sockets are resolved when all inputs have arrived."""
        inputs = {"lazy_var": [{"sender": "component1", "value": 42}, {"sender": "component2", "value": 43}]}
        assert are_all_lazy_variadic_sockets_resolved(complex_component, inputs) is True

    def test_lazy_variadic_sockets_partially_resolved(self, complex_component):
        """Missing some sender outputs means lazy variadic sockets are not resolved."""
        inputs = {
            "lazy_var": [{"sender": "component1", "value": 42}]  # Missing component2
        }
        assert are_all_lazy_variadic_sockets_resolved(complex_component, inputs) is False

    def test_lazy_variadic_sockets_with_no_inputs(self, complex_component):
        """No inputs: lazy variadic socket is not resolved."""
        inputs = {}
        assert are_all_lazy_variadic_sockets_resolved(complex_component, inputs) is False

    def test_lazy_variadic_sockets_with_predecessors_executed(self, complex_component):
        """
        Ensures that if all predecessors have executed (but produced no output),
        the lazy variadic socket is still considered resolved.
        """
        inputs = {
            "lazy_var": [
                {"sender": "component1", "value": _NO_OUTPUT_PRODUCED},
                {"sender": "component2", "value": _NO_OUTPUT_PRODUCED},
            ]
        }
        assert are_all_lazy_variadic_sockets_resolved(complex_component, inputs) is True


class TestGreedySocketReadiness:
    def test_greedy_socket_ready(self, complex_component):
        """A single valid input is enough for a greedy variadic socket to be considered ready."""
        inputs = {"greedy_var": [{"sender": "component1", "value": 42}]}
        assert is_any_greedy_socket_ready(complex_component, inputs) is True

    def test_greedy_socket_multiple_inputs_ready(self, complex_component):
        """Multiple valid inputs on a greedy socket is also fine—it's still ready."""
        inputs = {"greedy_var": [{"sender": "component1", "value": 42}, {"sender": "component2", "value": 43}]}
        assert is_any_greedy_socket_ready(complex_component, inputs) is True

    def test_greedy_socket_not_ready(self, complex_component):
        """If the only input is _NO_OUTPUT_PRODUCED, the greedy socket isn't ready."""
        inputs = {"greedy_var": [{"sender": "component1", "value": _NO_OUTPUT_PRODUCED}]}
        assert is_any_greedy_socket_ready(complex_component, inputs) is False

    def test_greedy_socket_no_inputs(self, complex_component):
        """No inputs at all: the greedy socket is not ready."""
        inputs = {}
        assert is_any_greedy_socket_ready(complex_component, inputs) is False

    def test_greedy_socket_with_user_input(self, complex_component):
        """User input can also trigger readiness for a greedy variadic socket."""
        inputs = {"greedy_var": [{"sender": None, "value": 42}]}
        assert is_any_greedy_socket_ready(complex_component, inputs) is True
