from v1.environment.controller import ControllerManual
from v2.environment.controller import ControllerQLearning
from v20.environment.controller import ControllerDDQL
from v24.environment.controller import ControllerPPONormalAD
from v3.environment.controller import ControllerAdvancedQLearning
from v4.environment.controller import ControllerCorpusQLearning
from v5.environment.controller import ControllerIdealADQLearning
from v6.environment.controller import ControllerSarsa
from v7.environment.controller import ControllerIdealADSarsa
from v8.environment.controller import ControllerOptimized
from v9.environment.controller import ControllerOptimizedQLearningAE
from v10.environment.controller import ControllerOptimizedQLearningIF
from v98.environment.controller import ControllerOneStepEpisodeQLearning
from v99.environment.controller import ControllerBruteForce
from environment.state_handling import get_prototype

CONTROLLER = None


def get_controller():
    global CONTROLLER
    if not CONTROLLER:
        proto = get_prototype()

        match proto:
            case "1":
                CONTROLLER = ControllerManual()
            case "2":
                CONTROLLER = ControllerQLearning()
            case "3":
                CONTROLLER = ControllerAdvancedQLearning()
            case "4":
                CONTROLLER = ControllerCorpusQLearning()
            case "5":
                CONTROLLER = ControllerIdealADQLearning()
            case "6":
                CONTROLLER = ControllerSarsa()
            case "7":
                CONTROLLER = ControllerIdealADSarsa()
            case "8":
                CONTROLLER = ControllerOptimized()
            case "9":
                CONTROLLER = ControllerOptimizedQLearningAE()
            case "10":
                CONTROLLER = ControllerOptimizedQLearningIF()
            case "98":
                CONTROLLER = ControllerOneStepEpisodeQLearning()
            case "99":
                CONTROLLER = ControllerBruteForce()


            case "20":
                CONTROLLER = ControllerDDQL()
            case "24":
                CONTROLLER = ControllerPPONormalAD()


            case _:
                print("WARNING: Unknown prototype. Falling back to default controller v1!")
                CONTROLLER = ControllerManual()
            
    return CONTROLLER
