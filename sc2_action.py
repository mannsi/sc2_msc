from pysc2.lib import actions
import constants


class Sc2Action:
    def __init__(self, internal_id, action_id, has_location, has_queued=True):
        self.internal_id = internal_id
        self.action_id = action_id
        self.has_location = has_location
        self.location = None
        self.has_queued = has_queued

    def set_location(self, location):
        self.location = location

    def get_function_call(self):
        """
        Get a function call expected by SC2Env.step function
        :param location: tuple of x, y coordinates. Can be skipped if there is no location needed
        :return: pysc2.lib.actions.FunctionCall object
        """
        args = []
        if self.has_queued:
            args.append(constants.NOT_QUEUED)
        if self.has_location:
            args.append(self.location)
        return actions.FunctionCall(self.action_id, args)

    def __str__(self):
        return self.internal_id




