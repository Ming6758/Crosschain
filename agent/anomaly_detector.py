# agent/anomaly_detector.py
from typing import Dict, Any

def check_anomaly(state: Dict[str, Any]) -> Dict[str, Any]:
    details_lower = state["details"].lower()
    is_dump = "dump" in details_lower or "transfer" in details_lower
    high_impact = state["tokenPriceImpact"] > 0.25
    new_wallet = len(state["walletHistory"]) < 5  # Example: If history short, flag as new/proxy wallet
    rapid_activity = any("recent_creation" in h.get("note", "") for h in state["walletHistory"])  # Placeholder for real analysis
    
    if is_dump or high_impact or new_wallet or rapid_activity:
        state["anomaly"] = True
    else:
        state["anomaly"] = False
    return state
