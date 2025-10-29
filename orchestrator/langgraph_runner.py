# orchestrator/langgraph_runner.py
from langgraph.graph import StateGraph
from typing import TypedDict, Optional, List
from agent.anomaly_detector import check_anomaly
from agent.exploit_classifier import classify
from agent.attack_matcher import match_attack
from agent.remediation_advisor import recommend_action
from agent.explainer_agent import explain_result

class AgentState(TypedDict, total=False):
    source: str
    alertType: str
    details: str
    timestamp: int
    chainId: int
    contractContext: str
    walletHistory: List[dict]
    tokenPriceImpact: float

    anomaly: Optional[bool]
    exploit_type: Optional[str]
    matching_attacks: Optional[List[dict]]
    remediation: Optional[str]
    final_explanation: Optional[str]

def create_graph():
    g = StateGraph(AgentState)

    g.add_node("AnomalyDetection", check_anomaly)
    g.add_node("ExploitClassification", classify)
    g.add_node("MatchAgainstKnownExploits", match_attack)
    g.add_node("SuggestRemediation", recommend_action)
    g.add_node("Explain", explain_result)

    g.set_entry_point("AnomalyDetection")
    g.add_edge("AnomalyDetection", "ExploitClassification")
    g.add_edge("ExploitClassification", "MatchAgainstKnownExploits")
    g.add_edge("MatchAgainstKnownExploits", "SuggestRemediation")
    g.add_edge("SuggestRemediation", "Explain")
    g.set_finish_point("Explain")

    return g.compile()
