from pysc2.lib import actions

_NOT_QUEUED = [0]


class ScAction:
    def __init__(self, action_id, has_location):
        self.action_id = action_id
        self.has_location = has_location

    def get_function_call(self, location=None):
        """
        Get a function call expected by SC2Env.step function
        :param location: tuple of x, y coordinates. Can be skipped if there is no location needed
        :return: pysc2.lib.actions.FunctionCall object
        """
        if self.has_location:
            return actions.FunctionCall(self.action_id, [_NOT_QUEUED, location])
        else:
            return actions.FunctionCall(self.action_id, [])




