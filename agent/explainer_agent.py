# agent/explainer_agent.py
from openai import OpenAI
from groq import Groq
from typing import Dict, Any
import os
from dotenv import load_dotenv

load_dotenv()

class ExplainerClient:
    def __init__(self):
        self.openai_client = OpenAI()
        self.groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    
    def explain_with_openai(self, prompt: str) -> str:
        completion = self.openai_client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2
        )
        return completion.choices[0].message.content.strip()
    
    def explain_with_groq(self, prompt: str) -> str:
        completion = self.groq_client.chat.completions.create(
            model="llama3-70b-8192",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2
        )
        return completion.choices[0].message.content.strip()

def explain_result(state: Dict[str, Any]) -> Dict[str, Any]:
    client = ExplainerClient()
    
    # Format matching attacks for display
    matched_attacks = "\n".join(
        f"- {attack['content'][:100]}... (Source: {attack['metadata'].get('source', 'unknown')})"
        for attack in state["matching_attacks"]
    ) if state.get("matching_attacks") else "No similar attacks found in database"
    
    prompt = f"""
    Prepare a security incident report for a blockchain security team:
    
    === INCIDENT DETAILS ===
    Chain: {state.get('chainId', 'Unknown')}
    Alert Type: {state.get('alertType', 'Unknown')}
    Exploit Classification: {state.get('exploit_type', 'Unknown')}
    
    === MATCHING HISTORICAL ATTACKS ===
    {matched_attacks}
    
    === RECOMMENDED ACTIONS ===
    {state.get('remediation', 'No specific recommendation')}
    
    Please provide:
    1. A concise summary of the incident
    2. Confidence level in the classification (High/Medium/Low)
    3. Immediate next steps
    4. Long-term mitigation suggestions
    """
    
    # Try OpenAI first, fallback to Groq
    try:
        state["final_explanation"] = client.explain_with_openai(prompt)
    except Exception as e:
        print(f"OpenAI failed, falling back to Groq: {str(e)}")
        state["final_explanation"] = client.explain_with_groq(prompt)
    
    return state