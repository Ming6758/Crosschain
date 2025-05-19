from pinecone import Pinecone
import os
from dotenv import load_dotenv
from pinecone import ServerlessSpec
from tqdm import tqdm
from openai import OpenAI


api_key = os.getenv("PINECONE_API_KEY")

# Configure client
pc = Pinecone(api_key=api_key)

index_name = 'crosschain'

# Instantiate an index client
index = pc.Index(name=index_name)

# View index stats of our new, empty index
print('Database stats: ',index.describe_index_stats())