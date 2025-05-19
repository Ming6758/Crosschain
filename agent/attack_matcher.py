# agent/attack_matcher.py
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings
from typing import Dict, Any
import os
from dotenv import load_dotenv

load_dotenv()

def match_attack(state: Dict[str, Any]) -> Dict[str, Any]:
    # Initialize embeddings - using your preferred approach
    embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")
    
    # Connect to Pinecone index using your working pattern
    vectorstore = PineconeVectorStore(
        index_name="crosschain",
        embedding=embeddings
    )
    
    # Create search query combining exploit type and details
    query = f"{state['exploit_type']}: {state['details']}"
    
    # Perform similarity search - using same approach as your RAG implementation
    docs = vectorstore.similarity_search(
        query=query,
        k=3  # Number of documents to retrieve
    )
    
    # Format results to include both content and metadata
    state["matching_attacks"] = [{
        "content": doc.page_content,
        "metadata": doc.metadata
    } for doc in docs]

    
    return state