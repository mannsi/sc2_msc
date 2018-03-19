import sys

from pysc2.lib import actions


def action_ids_to_action_names(action_ids):
    """
    Takes an iterable of action ids and returns an ordered list of corresponding action names
    :param action_ids: Iterable of action ids
    :return: List of action names
    """
    action_names = []

    for action_id in action_ids:
        action_name = actions.FUNCTIONS[action_id].name
        action_names.append(action_name)
    return action_names


def get_command_param_val(param_name, remove_from_params, default_val):
    # Hacky way to have file log level as input param. Could not figure out a nicer way without breaking the pysc2 logging.
    try:
        my_param_index = sys.argv.index(param_name)
    except ValueError:
        return default_val
    param_val = sys.argv[my_param_index + 1]
    if remove_from_params:
        sys.argv.pop(my_param_index + 1)
        sys.argv.pop(my_param_index)
    return param_val
