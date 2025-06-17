from environment.state_handling import get_prototype
from v20.environment.controller import ControllerDDQL
from v21.environment.controller import ControllerDDQLIdealAD
from v24.environment.controller import ControllerPPONormalAD
from v8.environment.controller import ControllerOptimized

CONTROLLER = None


def get_controller():
    global CONTROLLER
    if not CONTROLLER:
        proto = get_prototype()

        match proto:
            case "8":
                CONTROLLER = ControllerOptimized()

            case "20":
                CONTROLLER = ControllerDDQL()
            case "21":
                CONTROLLER = ControllerDDQLIdealAD()
            case "24":
                CONTROLLER = ControllerPPONormalAD()

            case _:
                print("WARNING: Unknown prototype. Falling back to default controller v1!")
                CONTROLLER = ControllerOptimized()

    return CONTROLLER
