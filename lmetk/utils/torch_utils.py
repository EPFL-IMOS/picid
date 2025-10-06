from typing import Any, Dict, OrderedDict


def correct_state_dict_keys(state_dict: Dict[str, Any]) -> OrderedDict:
    """Removes an extraneous 'backbone.' prefix from state_dict keys."""
    new_state_dict = OrderedDict()
    for key, value in state_dict.items():
        # Use removeprefix for a clean and explicit operation
        new_key = key.removeprefix("backbone.")
        new_state_dict[new_key] = value
    return new_state_dict
