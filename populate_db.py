# populate_db.py
import os
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_core.documents import Document

load_dotenv()

embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")
index_name = "crosschain"
vectorstore = PineconeVectorStore(index_name=index_name, embedding=embeddings)

# Real exploit examples from security reports (categorized)
exploits = [
    Document(
        page_content="Ronin Bridge Exploit: Hackers stole $625 million from Axie Infinity's Ronin Network by compromising validator nodes in a social engineering attack.",
        metadata={"source": "2022 Ronin Hack", "type": "Bridge Exploit"}
    ),
    Document(
        page_content="Poly Network Hack: Exploited a cross-chain message vulnerability, allowing the hacker to mint tokens on multiple chains, stealing $611 million (later returned).",
        metadata={"source": "2021 Poly Network", "type": "Cross-Chain Exploit"}
    ),
    Document(
        page_content="Beanstalk Governance Attack: Attacker used a flash loan to borrow governance tokens, pass a malicious proposal, and drain $182 million.",
        metadata={"source": "2022 Beanstalk", "type": "Governance Attack"}
    ),
    Document(
        page_content="Mango Markets Oracle Manipulation: Exploited price oracle by manipulating SOL price feeds, borrowing $116 million in a flash loan attack.",
        metadata={"source": "2022 Mango", "type": "Oracle Manipulation"}
    ),
    Document(
        page_content="Cream Finance Flash Loan: Multiple flash loans used to manipulate token prices and drain $130 million from lending pools.",
        metadata={"source": "2021 Cream Finance", "type": "Flash Loan"}
    ),
    Document(
        page_content="Ankr Exit Scam: Developers minted unlimited tokens via a bug, dumped them on exchanges, causing a rug pull worth millions.",
        metadata={"source": "2022 Ankr", "type": "Exit Scam"}
    ),
    # Add more as needed
]

# Upsert to Pinecone
vectorstore.add_documents(exploits)
print(f"Upserted {len(exploits)} exploit examples to Pinecone.")
