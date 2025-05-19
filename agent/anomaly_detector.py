# agent/anomaly_detector.py
from typing import Dict, Any

def check_anomaly(state: Dict[str, Any]) -> Dict[str, Any]:
    if "dump" in state["details"].lower() or state["tokenPriceImpact"] > 0.25:
        state["anomaly"] = True
    else:
        state["anomaly"] = False
    return state