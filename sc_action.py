from pysc2.lib import actions
import constants


class ScAction:
    def __init__(self, internal_id, action_id, has_location):
        self.internal_id = internal_id
        self.action_id = action_id
        self.has_location = has_location
        self.location = None

    def set_location(self, location):
        self.location = location

    def get_function_call(self):
        """
        Get a function call expected by SC2Env.step function
        :param location: tuple of x, y coordinates. Can be skipped if there is no location needed
        :return: pysc2.lib.actions.FunctionCall object
        """
        if self.has_location:
            return actions.FunctionCall(self.action_id, [constants.NOT_QUEUED, self.location])
        else:
            return actions.FunctionCall(self.action_id, [])




