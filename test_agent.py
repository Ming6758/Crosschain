# test_agent.py
import requests
import json

TEST_CASES = [
    {
        "source": "0x1234...",
        "alertType": "SuspiciousTransfer",
        "details": "High-value dump to CEX via proxy wallet",
        "timestamp": 1713170000,
        "tokenPriceImpact": 0.31,
        "walletHistory": [],
        "chainId": 1,
        "contractContext": "ERC20.transferFrom(proxyWallet, binance, 1000000)"
    },
    {
        "source": "0xabcd...",
        "alertType": "GovernanceProposal",
        "details": "Sudden large voting power delegation to new address",
        "timestamp": 1713171000,
        "tokenPriceImpact": 0.05,
        "walletHistory": [],
        "chainId": 42161,
        "contractContext": "Governance.delegate(newDelegate)"
    },
    {
        "source": "0xdef0...",
        "alertType": "OracleUpdate",
        "details": "Sudden price feed deviation detected",
        "timestamp": 1713172000,
        "tokenPriceImpact": 0.45,
        "walletHistory": [{"note": "recent_creation"}],
        "chainId": 1,
        "contractContext": "Oracle.updatePrice(manipulatedValue)"
    },
]

def test_agent():
    url = "http://localhost:8000/process"
    
    for test_case in TEST_CASES:
        print(f"\nTesting case: {test_case['alertType']}")
        response = requests.post(
            url,
            headers={"Content-Type": "application/json"},
            data=json.dumps(test_case)
        )
        
        if response.status_code == 200:
            result = response.json()
            print("\n=== FINAL RESULT ===")
            print(result.get("final_explanation", "No explanation generated"))
            print("\n=== PROCESSING METADATA ===")
            print(result.get("processing_metadata", {}))
        else:
            print(f"Error: {response.status_code} - {response.text}")

if __name__ == "__main__":
    test_agent()
