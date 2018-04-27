from pysc2.lib import actions
import constants


class Sc2Action:
    def __init__(self, internal_id):
        self.internal_id = internal_id
        self.location = None

    def set_location(self, location):
        self.location = location

    def get_function_call(self):
        """
        Get a function call expected by SC2Env.step function
        :return: pysc2.lib.actions.FunctionCall object
        """
        action_id, has_location, has_queued = self._map(self.internal_id)

        args = []
        if has_queued:
            args.append(constants.NOT_QUEUED)
        if has_location:
            args.append(self.location)
        return actions.FunctionCall(action_id, args)

    def __str__(self):
        return self.internal_id

    @staticmethod
    def _map(internal_id):
        """ Maps internal_id to other parameters """
        if internal_id == constants.NO_OP:
            has_location = False
            has_queued = False
            action_id = actions.FUNCTIONS.no_op.id
        elif internal_id == constants.MOVE_TO_ENEMY:
            has_location = True
            has_queued = True
            action_id = actions.FUNCTIONS.Move_screen.id
        elif internal_id == constants.MOVE_FROM_ENEMY:
            has_location = True
            has_queued = True
            action_id = actions.FUNCTIONS.Move_screen.id
        elif internal_id == constants.ATTACK_ENEMY:
            has_location = True
            has_queued = True
            action_id = actions.FUNCTIONS.Attack_screen.id
        elif internal_id == constants.FLIGHT:
            has_location = False
            has_queued = True
            action_id = actions.FUNCTIONS.Morph_VikingFighterMode_quick.id
        elif internal_id == constants.LAND:
            has_location = False
            has_queued = True
            action_id = actions.FUNCTIONS.Morph_VikingAssaultMode_quick.id
        else:
            raise ValueError(f"Received unknow internal action id {internal_id}")
        return action_id, has_location, has_queued


def action_id_to_internal_id(action_id):
    if action_id == actions.FUNCTIONS.no_op.id:
        return constants.NO_OP
    elif action_id == actions.FUNCTIONS.Move_screen.id:
        return constants.MOVE_TO_ENEMY
    elif action_id == actions.FUNCTIONS.Move_screen.id:
        return constants.MOVE_FROM_ENEMY
    elif action_id == actions.FUNCTIONS.Attack_screen.id:
        return constants.ATTACK_ENEMY
    elif action_id == actions.FUNCTIONS.Morph_VikingFighterMode_quick.id:
        return constants.FLIGHT
    elif action_id == actions.FUNCTIONS.Morph_VikingAssaultMode_quick.id:
        return constants.LAND
    else:
        raise ValueError(f"Received unknow action id {action_id}")


def internal_id_to_action_id(internal_id):
    if internal_id == constants.NO_OP:
        return actions.FUNCTIONS.no_op.id
    elif internal_id == constants.MOVE_TO_ENEMY:
        return actions.FUNCTIONS.Move_screen.id
    elif internal_id == constants.MOVE_FROM_ENEMY:
        return actions.FUNCTIONS.Move_screen.id
    elif internal_id == constants.ATTACK_ENEMY:
        return actions.FUNCTIONS.Attack_screen.id
    elif internal_id == constants.FLIGHT:
        return actions.FUNCTIONS.Morph_VikingFighterMode_quick.id
    elif internal_id == constants.LAND:
        return actions.FUNCTIONS.Morph_VikingAssaultMode_quick.id
    else:
        raise ValueError(f"Received unknow internal action id {internal_id}")
