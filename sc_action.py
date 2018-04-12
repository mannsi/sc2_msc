from pysc2.lib import actions
import constants


class ScAction:
    def __init__(self, internal_id, action_id, has_location):
        self.internal_id = internal_id
        self.action_id = action_id
        self.has_location = has_location

    def get_function_call(self, location=None):
        """
        Get a function call expected by SC2Env.step function
        :param location: tuple of x, y coordinates. Can be skipped if there is no location needed
        :return: pysc2.lib.actions.FunctionCall object
        """
        if self.has_location:
            return actions.FunctionCall(self.action_id, [constants.NOT_QUEUED, location])
        else:
            return actions.FunctionCall(self.action_id, [])




