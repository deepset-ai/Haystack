import pytest

from haystack.preview.components.generators.hf_utils import check_generation_params


@pytest.mark.unit
def test_empty_dictionary():
    # no exception raised
    check_generation_params({})


@pytest.mark.unit
def test_valid_generation_parameters():
    # these are valid parameters
    kwargs = {"max_new_tokens": 100, "temperature": 0.8}
    additional_accepted_params = None
    check_generation_params(kwargs, additional_accepted_params)


@pytest.mark.unit
def test_invalid_generation_parameters():
    # these are invalid parameters
    kwargs = {"invalid_param": "value"}
    additional_accepted_params = None
    with pytest.raises(ValueError):
        check_generation_params(kwargs, additional_accepted_params)


@pytest.mark.unit
def test_additional_accepted_params_empty_list():
    kwargs = {"temperature": 0.8}
    additional_accepted_params = []
    check_generation_params(kwargs, additional_accepted_params)


@pytest.mark.unit
def test_additional_accepted_params_non_empty_list():
    # both are valid parameters
    kwargs = {"temperature": 0.8}
    additional_accepted_params = ["max_new_tokens"]
    check_generation_params(kwargs, additional_accepted_params)


@pytest.mark.unit
def test_additional_accepted_params_valid_invalid():
    kwargs = {"temperature": 0.8}
    additional_accepted_params = ["valid_param"]
    # does not raise exception because valid_param is in additional_accepted_params
    check_generation_params(kwargs, additional_accepted_params)
