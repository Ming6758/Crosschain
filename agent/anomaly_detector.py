# agent/anomaly_detector.py
from typing import Dict, Any, List

def _is_new_wallet(history: List[dict]) -> bool:
    """Simple heuristic: <5 txs → treat as new / proxy wallet."""
    return len(history) < 5

def _has_rapid_activity(history: List[dict]) -> bool:
    """Flag if any entry mentions recent creation or burst."""
    return any("recent" in h.get("note", "").lower() for h in history)

def check_anomaly(state: Dict[str, Any]) -> Dict[str, Any]:
    details = state["details"].lower()
    high_impact = state.get("tokenPriceImpact", 0) > 0.25
    dump_keywords = {"dump", "transfer", "withdraw", "liquidate"}
    is_dump = any(k in details for k in dump_keywords)

    state["anomaly"] = (
        is_dump
        or high_impact
        or _is_new_wallet(state.get("walletHistory", []))
        or _has_rapid_activity(state.get("walletHistory", []))
    )
    return state
