# agent/remediation_advisor.py
from typing import Dict, Any

def recommend_action(state: Dict[str, Any]) -> Dict[str, Any]:
    exploit_type = state["exploit_type"].lower()
    
    if "flash loan" in exploit_type:
        remediation = "Pause lending pool + notify DAO"
    elif "exit scam" in exploit_type:
        remediation = "Freeze treasury multisig, alert community"
    elif "oracle manipulation" in exploit_type:
        remediation = "Pause price feeds, switch to backup oracle"
    elif "token dump" in exploit_type:
        remediation = "Monitor CEX inflows, consider blacklisting"
    elif "governance attack" in exploit_type:
        remediation = "Freeze governance proposals, emergency vote"
    else:
        remediation = "Escalate to core security team"
    
    state["remediation"] = remediation
    return state