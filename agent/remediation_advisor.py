# agent/remediation_advisor.py
from typing import Dict, Any

def recommend_action(state: Dict[str, Any]) -> Dict[str, Any]:
    exploit_type = state["exploit_type"].lower()
    
    if "flash loan" in exploit_type:
        remediation = "Pause lending pools, blacklist suspicious addresses, and notify DAO for audit."
    elif "exit scam" in exploit_type or "rug pull" in exploit_type:
        remediation = "Freeze treasury multisig, alert community via social channels, and pursue legal action."
    elif "oracle manipulation" in exploit_type:
        remediation = "Pause price feeds, switch to backup oracles, and validate data sources."
    elif "token dump" in exploit_type:
        remediation = "Monitor CEX inflows, consider temporary trading halts, and blacklist dump wallets."
    elif "governance attack" in exploit_type:
        remediation = "Freeze governance proposals, initiate emergency vote, and review delegation mechanics."
    elif "bridge exploit" in exploit_type:
        remediation = "Halt bridge operations, audit cross-chain messages, and recover funds if possible."
    elif "smart contract vulnerability" in exploit_type:
        remediation = "Pause contract, deploy patched version, and compensate affected users."
    else:
        remediation = "Escalate to core security team for manual review and containment."
    
    state["remediation"] = remediation
    return state
