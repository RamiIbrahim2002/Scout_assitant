import os
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI

load_dotenv()
def load_rag_pipeline():
    # Use OpenAI embeddings to match the original vector DB creation
    embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")
    
    # Load the vector database
    db = FAISS.load_local("stellar_vector_db", embeddings, allow_dangerous_deserialization=True)
    
    # Set up retriever with an increased number of documents to retrieve (k=5)
    retriever = db.as_retriever(search_kwargs={"k": 5})
    
    # Define prompt template
    template = """You are a helpful Discord support agent. Follow these guidelines:
1. Provide relevant information     
2. Provide clear and concise technical assistance
3. If unsure, say "I don't have enough information to answer confidently"

Sources:
{context}

Question: {question}
Answer:"""
    prompt = ChatPromptTemplate.from_template(template)
    
    # Load LLM with hardcoded API key (TEMPORARY FOR TESTING ONLY)
    llm = ChatOpenAI(
        model="gpt-3.5-turbo", 
        temperature=0.3
    )
    
    # Define a function to build and log the context
    def build_context(inp):
        # Handle different input types robustly
        if isinstance(inp, dict):
            question = inp.get("question", "")
        elif isinstance(inp, str):
            question = inp
        else:
            question = ""
        
        # Retrieve relevant documents from the FAISS index
        docs = retriever.get_relevant_documents(question)
        
        # Build the context string by joining the content of each document
        context = " ".join(doc.page_content for doc in docs)
        
        # Log the retrieved context
        print("Retrieved context for question '{}':\n{}\n".format(question, context))
        
        # Return the updated input dictionary
        return {"question": question, "context": context}
    
    # Create the RAG chain by chaining the steps
    rag_chain = (
        RunnablePassthrough()
        | build_context
        | prompt
        | llm
    )
    return rag_chain

def test_rag_pipeline():
    # Load the RAG chain
    rag_chain = load_rag_pipeline()
    
    # Test cases covering various Discord support scenarios
    test_questions = [
        "how do i add proxy",
    "what is proxy",
    "what is akamai",
    "what is px",
    "how can i generate shape cookies"
    ]
    
    # Run tests and print the output for each test
    for question in test_questions:
        response = rag_chain.invoke(question)
        print(f"Question: {question}")
        print(f"Answer: {response.content}\n{'='*50}\n")

if __name__ == "__main__":
    test_rag_pipeline()