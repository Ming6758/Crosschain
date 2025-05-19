from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
import os
from dotenv import load_dotenv
import tiktoken

load_dotenv()

# Initialize components
embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")
llm = ChatOpenAI(model="gpt-4", temperature=0)

# Connect to Pinecone index
index_name = 'crosschain'
vectorstore = PineconeVectorStore(
    index_name=index_name,
    embedding=embeddings
)

# Initialize tokenizer for counting tokens
tokenizer = tiktoken.get_encoding("cl100k_base")

def count_tokens(text: str) -> int:
    """Count tokens in text using tiktoken"""
    return len(tokenizer.encode(text))



def rag(query: str, retrieval_count: int = 3, token_limit: int = 8000, verbose: bool = False):
    """
    Enhanced RAG function with configurable retrieval and token limits
    
    Args:
        query: The question to answer
        retrieval_count: Number of documents to retrieve
        token_limit: Maximum allowed tokens for context+question
        verbose: Whether to print debug information
    
    Returns:
        str: The generated answer
    """
    # Single unified prompt template
    PROMPT_TEMPLATE = """Answer the question based on the context below. If you don't know the answer, just say you don't know.

    Context:
    {context}

    Question: {question}
    Answer:"""
        
    # Initialize retriever with custom search count
    retriever = vectorstore.as_retriever(search_kwargs={"k": retrieval_count})
    
    # Retrieve documents
    docs = retriever.invoke(query)
    
    if verbose:
        print(f"Retrieved {len(docs)} documents")
    
    # Calculate base prompt tokens (without context)
    base_prompt = PROMPT_TEMPLATE.format(context="", question=query)
    base_tokens = count_tokens(base_prompt)
    remaining_tokens = token_limit - base_tokens
    
    if remaining_tokens <= 0:
        return "Token limit too low for base prompt"
    
    # Build context within token limits
    context_parts = []
    current_tokens = 0
    
    for doc in docs:
        doc_content = doc.page_content
        doc_tokens = count_tokens(doc_content)
        
        if current_tokens + doc_tokens > remaining_tokens:
            if verbose:
                print(f"Skipping document - would exceed token limit")
            break
            
        context_parts.append(doc_content)
        current_tokens += doc_tokens
    
    if not context_parts:
        return "No relevant context found within token limits"
    
    # Create the final context
    context = "\n\n---\n\n".join(context_parts)
    
    # Construct the full prompt
    full_prompt = PROMPT_TEMPLATE.format(context=context, question=query)
    
    if verbose:
        print("\n" + "="*50 + " FULL PROMPT " + "="*50)
        print(full_prompt)
    
    # Create the QA chain
    PROMPT = PromptTemplate(
        template=PROMPT_TEMPLATE, 
        input_variables=["context", "question"]
    )
    
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        chain_type_kwargs={"prompt": PROMPT},
        return_source_documents=False
    )
    
    # Execute the chain
    try:
        result = qa_chain.invoke({"query": query, "context": context})
        return result["result"]
    except Exception as e:
        if verbose:
            print(f"Error during generation: {str(e)}")
        return "Sorry, I encountered an error while generating the answer."


# Example usage
query = (
    "Which training method should I use for sentence transformers when I only have pairs of related sentences?"
)




# With custom parameters
print("\nCustom settings (5 retrievals, 4000 token limit):")
print(rag(query, retrieval_count=5, token_limit=8000, verbose=True))