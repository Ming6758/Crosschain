# test_agent.py
import requests, json, time
from datetime import datetime

TEST_CASES = [
    # original
    {
        "source": "0x1234...",
        "alertType": "SuspiciousTransfer",
        "details": "High-value dump to CEX via proxy wallet",
        "timestamp": int(datetime.now().timestamp()),
        "tokenPriceImpact": 0.31,
        "walletHistory": [],
        "chainId": 1,
        "contractContext": "ERC20.transferFrom(proxy, binance, 1000000)"
    },
    # governance
    {
        "source": "0xabcd...",
        "alertType": "GovernanceProposal",
        "details": "Sudden large voting power delegation to new address",
        "timestamp": int(datetime.now().timestamp()),
        "tokenPriceImpact": 0.05,
        "walletHistory": [{"note": "recent_creation"}],
        "chainId": 42161,
        "contractContext": "Governance.delegate(newDelegate)"
    },
    # oracle
    {
        "source": "0xdef0...",
        "alertType": "OracleUpdate",
        "details": "Price feed deviation of 45% in 2 blocks",
        "timestamp": int(datetime.now().timestamp()),
        "tokenPriceImpact": 0.45,
        "walletHistory": [],
        "chainId": 1,
        "contractContext": "Oracle.updatePrice(manipulatedValue)"
    },
]

def test_agent():
    url = "http://localhost:8000/process"
    for i, case in enumerate(TEST_CASES, 1):
        print(f"\n=== TEST {i}: {case['alertType']} ===")
        t0 = time.time()
        r = requests.post(url, json=case, timeout=10)
        elapsed = time.time() - t0
        if r.status_code == 200:
            out = r.json()
            print("FINAL REPORT:\n", out.get("final_explanation"))
            print("\nMETADATA:", out.get("processing_metadata"))
            print(f"Latency: {elapsed:.2f}s")
        else:
            print(f"ERROR {r.status_code}: {r.text}")

if __name__ == "__main__":
    test_agent()
