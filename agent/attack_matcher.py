# agent/attack_matcher.py
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings
from typing import Dict, Any
import os
from dotenv import load_dotenv

load_dotenv()

embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")
vectorstore = PineconeVectorStore(
    index_name="crosschain",
    embedding=embeddings,
    namespace="exploits"          # optional isolation
)

def match_attack(state: Dict[str, Any]) -> Dict[str, Any]:
    query = f"{state.get('exploit_type', '')}: {state['details']}"
    try:
        docs = vectorstore.similarity_search(query, k=4)   # 4 for richer context
    except Exception as e:
        print(f"Pinecone search failed: {e}")
        docs = []

    state["matching_attacks"] = [
        {"content": d.page_content, "metadata": d.metadata} for d in docs
    ]
    return state
