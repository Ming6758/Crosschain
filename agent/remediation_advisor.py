# agent/remediation_advisor.py
from typing import Dict, Any

REMEDIATION_MAP = {
    "flash loan": "Pause lending pools, blacklist attacker contracts, notify DAO.",
    "governance": "Freeze proposals, emergency vote, audit delegation logic.",
    "exit scam": "Freeze treasury multisig, broadcast community alert, pursue legal.",
    "oracle manipulation": "Pause price feeds, switch to backup oracle, validate sources.",
    "token dump": "Monitor CEX inflows, consider temporary trade halt, blacklist wallets.",
    "bridge exploit":Finland "Halt bridge, audit cross-chain messages, attempt fund recovery.",
    "smart contract vulnerability": "Pause contract, deploy patched version, compensate users.",
}

def recommend_action(state: Dict[str, Any]) -> Dict[str, Any]:
    et = state.get("exploit_type", "").lower()
    remediation = next(
        (v for k, v in REMEDIATION_MAP.items() if k in et),
        "Escalate to core security team for manual triage."
    )
    state["remediation"] = remediation
    return state
