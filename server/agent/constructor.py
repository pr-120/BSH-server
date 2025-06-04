from agent.agent_representation import AgentRepresentation
from environment.state_handling import get_prototype
from v1.agent.agent import AgentManual
from v2.agent.agent import AgentQLearning
from v3.agent.agent import AgentAdvancedQLearning
from v4.agent.agent import AgentCorpusQLearning
from v5.agent.agent import AgentIdealADQLearning
from v6.agent.agent import AgentSarsa
from v7.agent.agent import AgentIdealADSarsa
from v8.agent.agent import AgentOptimized
from v9.agent.agent import AgentOptimizedQLearningAE
from v10.agent.agent import AgentOptimizedQLearningIF
from v98.agent.agent import AgentOneStepEpisodeQLearning
from v99.agent.agent import AgentBruteForce


from v20.agent.agent import AgentDDQL
from v21.agent.agent import AgentDDQLIdealAD
from v24.agent.agent import AgentPPONormalAD
from v25.agent.agent import AgentPPOIdealAD

AGENT = None


def get_agent():
    global AGENT
    if not AGENT:
        proto = get_prototype()

        match proto:

            case "1":
                AGENT = AgentManual()
            case "2":
                AGENT = AgentQLearning()
            case "3":
                AGENT = AgentAdvancedQLearning()
            case "4":
                AGENT = AgentCorpusQLearning()
            case "5":
                AGENT = AgentIdealADQLearning()
            case "6":
                AGENT = AgentSarsa()
            case "7":
                AGENT = AgentIdealADSarsa()
            case "8":
                AGENT = AgentOptimized()
            case "9":
                AGENT = AgentOptimizedQLearningAE()
            case "10":
                AGENT = AgentOptimizedQLearningIF()
            case "98":
                AGENT = AgentOneStepEpisodeQLearning()
            case "99":
                AGENT = AgentBruteForce()

            case "20":
                AGENT = AgentDDQL()
            case "21":
                AGENT = AgentDDQLIdealAD()
            case "24":
                AGENT = AgentPPONormalAD()
            case "25":
                AGENT = AgentPPOIdealAD()

            case _:
                print("WARNING: Unknown prototype. Falling back to default agent v1!")
                AGENT = AgentManual()

    return AGENT


def build_agent_from_repr(representation):
    assert isinstance(representation, AgentRepresentation)
    proto = get_prototype()

    match proto:
        case "1":
            print("WARNING: Agent v1 does not support building from representation! Returning fresh agent instance...")
            AGENT = AgentManual()
        case "2":
            AGENT = AgentQLearning(representation)
        case "3":
            AGENT = AgentAdvancedQLearning(representation)
        case "4":
            AGENT = AgentCorpusQLearning(representation)
        case "5":
            AGENT = AgentIdealADQLearning(representation)
        case "6":
            AGENT = AgentSarsa(representation)
        case "7":
            AGENT = AgentIdealADSarsa(representation)
        case "8":
            AGENT = AgentOptimized(representation)
        case "9":
            AGENT = AgentOptimizedQLearningAE(representation)
        case "10":
            AGENT = AgentOptimizedQLearningIF(representation)
        case "98":
            AGENT = AgentOneStepEpisodeQLearning(representation)
        case "99":
            print("WARNING: Agent v99 does not support building from representation! Returning fresh agent instance...")
            AGENT = AgentBruteForce()

        case "20":
            AGENT = AgentDDQL(representation)
        case "24":
            AGENT = AgentPPONormalAD(representation)
        case "25":
            AGENT = AgentPPOIdealAD(representation)

        case _:
            print("WARNING: Unknown prototype. Falling back to default agent v1!")
            AGENT = AgentManual()

    return AGENT
