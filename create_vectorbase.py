from pinecone_datasets import load_dataset
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings
import os
from dotenv import load_dotenv
from langchain.schema import Document
from tqdm.auto import tqdm
from typing import List

load_dotenv('/Users/minglin/Downloads/Langchain_RAG/.env')


# Load dataset with progress bar
print("Loading dataset...")
dataset = load_dataset('youtube-transcripts-text-embedding-ada-002', split='train[:10]')
print(dataset.head())
print(dataset.documents.iloc[0])


# Initialize OpenAI embeddings
embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")


def get_text_embedding(text: str) -> List[float]:
    """
    Generate embedding for a given text using OpenAI's embeddings model.
    """
    embedding_vector = embeddings.embed_query(text)
    return embedding_vector

#print(get_text_embedding('Generate embedding for a given text using OpenAIs embeddings model.'))
#print('1')



# Prepare documents for Pinecone with progress bar
print("Processing documents...")
docs = []
for _, row in tqdm(dataset.documents.iterrows(), total=len(dataset.documents), desc="Processing"):
    # Extract text from blob if it exists, otherwise use empty string
    text = row['blob'].get('text', '') if isinstance(row['blob'], dict) else ''

    print(text)
    
    
    # Create metadata dictionary excluding the 'text' field
    metadata = row['blob'].copy() if isinstance(row['blob'], dict) else {}
    if 'text' in metadata:
        del metadata['text']
    
    print(metadata)
    # Create LangChain Document
    docs.append(Document(
        page_content=text,
        metadata=metadata
    ))

print('11111')
print('111111')

# Create Pinecone vector store with batched uploads
index_name = 'crosschain'
batch_size = 100  # Adjust this based on your document sizes

# Initialize the vector store
print("Initializing Pinecone index...")
vectorstore = PineconeVectorStore.from_documents(
    documents=[],  # Start with empty list
    embedding=embeddings,
    index_name=index_name
)

# Upload documents in batches with progress bar
print("Uploading documents to Pinecone...")
for i in tqdm(range(0, len(docs), batch_size), desc="Uploading batches"):
    batch = docs[i:i + batch_size]
    try:
        vectorstore.add_documents(batch)
    except Exception as e:
        print(f"\nError uploading batch {i//batch_size + 1}: {str(e)}")
        # Optionally: save failed batch to retry later
        continue








print("\nVector database created successfully!")
print(f"Total documents uploaded: {len(docs)}")
