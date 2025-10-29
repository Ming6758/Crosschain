# agent/explainer_agent.py
from openai import OpenAI
from groq import Groq
from typing import Dict, Any
import os
from dotenv import load_dotenv

load_dotenv()

class Explainer:
    def __init__(self):
        self.openai = OpenAI()
        self.groq = Groq(api_key=os.getenv("GROQ_API_KEY"))

    def _call(self, prompt: str, model: str) -> str:
        return self.openai.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
        ).choices[0].message.content.strip()

    def openai(self, prompt: str) -> str:
        return self._call(prompt, "gpt-4o")

    def groq(self, prompt: str) -> str:
        return self.groq.chat.completions.create(
            model="llama3-70b-8192",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
        ).choices[0].message.content.strip()

def explain_result(state: Dict[str, Any]) -> Dict[str, Any]:
    client = Explainer()

    matches = "\n".join(
        f"- {a['content'][:120]}… (src: {a['metadata'].get('source','?')})"
        for a in state.get("matching_attacks", [])
    ) or "No historical matches."

    prompt = f"""
Write a **concise security incident report** (max 5 sentences) for the on-call team:

Chain: {state.get('chainId')}
Alert: {state.get('alertType')}
Classification: {state.get('exploit_type')}
Confidence: {'High' if state.get('anomaly') else 'Medium'}

Historical matches:
{matches}

Recommended action:
{state.get('remediation')}

Provide:
1. One-sentence summary
2. Confidence (High/Medium/Low)
3. Immediate next step
4. Long-term mitigation
"""

    try:
        state["final_explanation"] = client.openai(prompt)
    except Exception as e:
        print(f"OpenAI explainer failed → Groq: {e}")
        state["final_explanation"] = client.groq(prompt)
    return state
