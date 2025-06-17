from agent.agent_representation import AgentRepresentation
from agent.agent_representation_mutlilayer import AgentRepresentationMultiLayer
from environment.state_handling import get_prototype
from v20.agent.agent import AgentDDQL
from v21.agent.agent import AgentDDQLIdealAD
from v24.agent.agent import AgentPPONormalAD
from v8.agent.agent import AgentOptimized

AGENT = None


def get_agent():
    global AGENT
    if not AGENT:
        proto = get_prototype()

        match proto:

            case "8":
                AGENT = AgentOptimized()

            case "20":
                AGENT = AgentDDQL()
            case "21":
                AGENT = AgentDDQLIdealAD()
            case "24":
                AGENT = AgentPPONormalAD()

            case _:
                print("WARNING: Unknown prototype. Falling back to default agent v1!")
                AGENT = AgentOptimized()

    return AGENT


def build_agent_from_repr(representation):
    proto = get_prototype()

    if proto == "21":
        assert isinstance(representation,
                          AgentRepresentationMultiLayer), "Expected AgentRepresentationMultiLayer for proto 21"
        AGENT = AgentDDQLIdealAD(representation)
    else:
        assert isinstance(representation, AgentRepresentation), "Expected AgentRepresentation for other versions"

        match proto:

            case "8":
                AGENT = AgentOptimized(representation)
            case "20":
                AGENT = AgentDDQL(representation)
            case "24":
                AGENT = AgentPPONormalAD(representation)

            case _:
                print("WARNING: Unknown prototype. Falling back to default agent v1!")
                AGENT = AgentOptimized(representation)

    return AGENT
