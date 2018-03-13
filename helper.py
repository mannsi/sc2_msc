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
