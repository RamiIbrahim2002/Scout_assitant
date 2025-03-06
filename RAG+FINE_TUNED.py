import os
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from transformers import AutoTokenizer, AutoModelForCausalLM

load_dotenv()

# Load your fine-tuned Llama model and tokenizer from Hugging Face Hub.
llama_model_path = "RamiIbrahim/lora_model"  # Replace with your Hugging Face repository name
llama_tokenizer = AutoTokenizer.from_pretrained(llama_model_path)
llama_model = AutoModelForCausalLM.from_pretrained(llama_model_path)

def load_rag_pipeline():
    # Use OpenAI embeddings (ensure it matches your vector DB creation)
    embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")
    
    # Load the FAISS vector database (this remains local unless you host it elsewhere)
    db = FAISS.load_local("stellar_vector_db", embeddings, allow_dangerous_deserialization=True)
    
    # Set up the retriever (retrieving top 5 documents)
    retriever = db.as_retriever(search_kwargs={"k": 5})
    
    # Define your prompt template for the Discord support agent
    template = """You are a helpful Discord support agent. Follow these guidelines:
1. Provide relevant information     
2. Provide clear and concise technical assistance
3. If unsure, say "I don't have enough information to answer confidently"

Sources:
{context}

Question: {question}
Answer:"""
    prompt = ChatPromptTemplate.from_template(template)
    
    # Function to build context: retrieve documents and add them to the input
    def build_context(inp):
        # Handle input types (str or dict)
        if isinstance(inp, dict):
            question = inp.get("question", "")
        elif isinstance(inp, str):
            question = inp
        else:
            question = ""
        
        # Retrieve relevant documents from FAISS
        docs = retriever.get_relevant_documents(question)
        # Build a context string from the retrieved documents
        context = " ".join(doc.page_content for doc in docs)
        
        # Debug: log the retrieved context
        print("Retrieved context for question '{}':\n{}\n".format(question, context))
        
        # Return the updated input for the prompt template
        return {"question": question, "context": context}
    
    # Function that uses your fine-tuned Llama model for generation
    def llama_inference(prompt_input):
        """
        Expects a prompt string (or a dict containing the prompt text) from the previous step.
        Generates an answer using the fine-tuned Llama model.
        """
        # If prompt_input is a dict, extract the formatted text
        if isinstance(prompt_input, dict):
            prompt_text = prompt_input.get("text", "")
        else:
            prompt_text = prompt_input
        
        # Debug: print the final prompt
        print("Final prompt sent to Llama model:\n", prompt_text)
        
        # Tokenize the prompt and generate an answer
        inputs = llama_tokenizer(prompt_text, return_tensors="pt")
        output = llama_model.generate(inputs["input_ids"], max_length=512, do_sample=True)
        answer = llama_tokenizer.decode(output[0], skip_special_tokens=True)
        # Return the answer in a dict so downstream steps can access it using the "content" key
        return {"content": answer}
    
    # Build the RAG chain by piping together the steps:
    # 1. Pass the raw input
    # 2. Build context using the retriever
    # 3. Format the prompt with the context and question
    # 4. Generate an answer using the fine-tuned Llama model
    rag_chain = (
        RunnablePassthrough()
        | build_context
        | prompt
        | llama_inference
    )
    return rag_chain

def test_rag_pipeline():
    # Load the pipeline
    rag_chain = load_rag_pipeline()
    
    # Test questions for various support scenarios
    test_questions = [
        "how do i add proxy",
        "what is proxy",
        "what is akamai",
        "what is px",
        "how can i generate shape cookies"
    ]
    
    # Run tests and print output for each question
    for question in test_questions:
        response = rag_chain.invoke(question)
        print(f"Question: {question}")
        print(f"Answer: {response['content']}\n{'='*50}\n")

if __name__ == "__main__":
    test_rag_pipeline()
