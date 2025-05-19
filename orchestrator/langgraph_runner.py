# orchestrator/langgraph_runner.py
from langgraph.graph import StateGraph
from typing import TypedDict, Optional, List
from agent.anomaly_detector import check_anomaly
from agent.exploit_classifier import classify
from agent.attack_matcher import match_attack
from agent.remediation_advisor import recommend_action
from agent.explainer_agent import explain_result

class AgentState(TypedDict):
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
    # Define the graph with state schema
    workflow = StateGraph(AgentState)

    # Register nodes
    workflow.add_node("AnomalyDetection", check_anomaly)
    workflow.add_node("ExploitClassification", classify)
    workflow.add_node("MatchAgainstKnownExploits", match_attack)
    workflow.add_node("SuggestRemediation", recommend_action)
    workflow.add_node("Explain", explain_result)

    # Define the flow
    workflow.set_entry_point("AnomalyDetection")
    
    # Main flow
    workflow.add_edge("AnomalyDetection", "ExploitClassification")
    workflow.add_edge("ExploitClassification", "MatchAgainstKnownExploits")
    workflow.add_edge("MatchAgainstKnownExploits", "SuggestRemediation")
    workflow.add_edge("SuggestRemediation", "Explain")

    # Set the exit point (no need for explicit END node)
    workflow.set_finish_point("Explain")

    return workflow.compile()