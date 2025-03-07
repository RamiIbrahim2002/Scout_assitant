import os
import re
import json
import glob
import PyPDF2
import traceback
import logging
from tqdm import tqdm
from dotenv import load_dotenv

import discord
from discord.ext import commands

# LangChain and related libraries
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.schema import BaseRetriever

from pydantic import Field
from typing import Any, List

# Load environment variables
load_dotenv()

# Set up logging to both console and file.
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler("bot.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

###############################################################
#                   PDF & CONVERSATION PROCESSING             #
###############################################################

# Global configuration: use environment variable for your API key.
openai_api_key = os.getenv("OPENAI_API_KEY")
pdf_path = "Stellar.pdf"          # Path to your PDF file
conversation_dir = "data"         # Directory containing conversation JSON files
batch_size = 100                  # Process conversation files in batches

def extract_text_from_pdf(pdf_path):
    """Extract text from a single PDF file."""
    text = ""
    try:
        with open(pdf_path, "rb") as file:
            reader = PyPDF2.PdfReader(file)
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
        return text
    except Exception as e:
        logger.error("Error extracting text from PDF %s: %s", pdf_path, str(e))
        return ""

def clean_text(text):
    """Clean extracted text by removing URLs, reference markers, and extra whitespace."""
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'\[\d+\]', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def process_pdf_document(pdf_path):
    """Process PDF document and create a vector database."""
    logger.info("Processing PDF document: %s", pdf_path)
    raw_text = extract_text_from_pdf(pdf_path)
    cleaned_text = clean_text(raw_text)
    
    if len(cleaned_text) < 100:
        logger.warning("Not enough content was extracted from the PDF.")
        return None

    # Split text into chunks using a recursive splitter.
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separators=["\n\n", "\n", ". "]
    )
    chunks = text_splitter.split_text(cleaned_text)
    
    documents = [
        Document(
            page_content=chunk,
            metadata={
                "source": os.path.basename(pdf_path),
                "source_type": "documentation"
            }
        )
        for chunk in chunks
    ]
    
    logger.info("Creating vector database from %d document chunks...", len(documents))
    db = FAISS.from_documents(documents, OpenAIEmbeddings(model="text-embedding-ada-002", openai_api_key=openai_api_key))
    db.save_local("stellar_vector_db")
    logger.info("Document vector database saved as 'stellar_vector_db'")
    
    return db

def process_conversation_file(file_path):
    """Process a single conversation file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            conversation = json.load(f)
        if not conversation or len(conversation) < 2:
            return None
        processed_convo = process_single_conversation(conversation, os.path.basename(file_path))
        return processed_convo
    except Exception as e:
        logger.error("Error processing %s: %s", file_path, str(e))
        return None

def process_single_conversation(conversation, file_id):
    """Process a single conversation into a meaningful exchange."""
    filtered_msgs = [msg for msg in conversation 
                     if msg.get("content") and isinstance(msg.get("content"), str)
                     and not (msg.get("content", "").startswith("<@") and len(msg.get("content", "")) < 30)]
    
    if len(filtered_msgs) < 2:
        return None

    topics = []
    for msg in filtered_msgs:
        if msg.get("role") == "client" and len(msg.get("content", "")) > 15:
            potential_topics = ["amazon", "target", "checkout", "captcha", "proxy", "login", 
                                "bot", "stellar", "error", "issue", "problem", "setup"]
            for topic in potential_topics:
                if topic.lower() in msg.get("content", "").lower() and topic not in topics:
                    topics.append(topic)
    
    convo_text = f"File: {file_id}\nTopics: {', '.join(topics) if topics else 'general'}\n\n"
    for msg in filtered_msgs:
        role = msg.get("role", "unknown").upper()
        content = msg.get("content", "")
        convo_text += f"{role}: {content}\n\n"
    
    has_resolution = any(
        (msg.get("role") == "client" and "thank" in msg.get("content", "").lower()) or
        (msg.get("role") == "staff" and any(word in msg.get("content", "").lower() 
                                            for word in ["solved", "fixed", "works", "resolved", "closing"]))
        for msg in filtered_msgs
    )
    
    return {
        "text": convo_text.strip(),
        "topics": topics,
        "resolved": has_resolution,
        "file_id": file_id
    }

def batch_process_conversations(directory, batch_size=100):
    """Process conversation files in batches."""
    file_paths = glob.glob(os.path.join(directory, "**/*processed_messages.json"), recursive=True)
    logger.info("Found %d conversation files", len(file_paths))
    
    all_documents = []
    total_batches = (len(file_paths) + batch_size - 1) // batch_size
    
    for batch_idx in range(total_batches):
        start_idx = batch_idx * batch_size
        end_idx = min((batch_idx + 1) * batch_size, len(file_paths))
        batch_files = file_paths[start_idx:end_idx]
        
        logger.info("Processing batch %d/%d (%d files)", batch_idx + 1, total_batches, len(batch_files))
        batch_convos = []
        for file_path in tqdm(batch_files):
            processed = process_conversation_file(file_path)
            if processed:
                batch_convos.append(processed)
        
        batch_documents = []
        for convo in batch_convos:
            doc = Document(
                page_content=convo["text"],
                metadata={
                    "source": "conversation",
                    "file_id": convo["file_id"],
                    "topics": convo["topics"],
                    "resolved": convo["resolved"],
                    "source_type": "conversation"
                }
            )
            batch_documents.append(doc)
        
        all_documents.extend(batch_documents)
        logger.info("Processed %d conversations in batch %d", len(batch_documents), batch_idx + 1)
    
    logger.info("Total conversations processed: %d", len(all_documents))
    return all_documents

def create_conversation_db(conversation_dir, batch_size):
    """Create a vector database from all conversation files."""
    documents = batch_process_conversations(conversation_dir, batch_size)
    if not documents:
        logger.warning("No valid conversations found.")
        return None
    
    logger.info("Creating vector database from conversations...")
    embeddings = OpenAIEmbeddings(model="text-embedding-ada-002", openai_api_key=openai_api_key)
    convo_db = FAISS.from_documents(documents, embeddings)
    convo_db.save_local("stellar_conversation_db")
    logger.info("Created conversation database with %d entries", len(documents))
    return convo_db

###############################################################
#                     HYBRID RETRIEVER SETUP                  #
###############################################################

class HybridRetriever(BaseRetriever):
    doc_retriever: Any = Field(..., description="Retriever for documents")
    convo_retriever: Any = Field(..., description="Retriever for conversations")
    doc_weight: float = Field(0.7, description="Weight for the document retriever")
    k: int = Field(5, description="Total number of results to retrieve")
    
    def __init__(self, doc_retriever: Any, convo_retriever: Any, doc_weight: float = 0.7, k: int = 5):
        super().__init__(doc_retriever=doc_retriever, convo_retriever=convo_retriever, doc_weight=doc_weight, k=k)
        object.__setattr__(self, "convo_weight", 1.0 - doc_weight)
    
    def get_relevant_documents(self, query: str) -> List[Any]:
        doc_k = max(1, int(self.k * self.doc_weight))
        convo_k = max(1, self.k - doc_k)
        doc_results = self.doc_retriever.get_relevant_documents(query, k=doc_k) if self.doc_retriever else []
        convo_results = self.convo_retriever.get_relevant_documents(query, k=convo_k) if self.convo_retriever else []
        return doc_results + convo_results

def setup_full_rag_system():
    """Process documents and conversations, and set up the hybrid RAG system."""
    embeddings = OpenAIEmbeddings(model="text-embedding-ada-002", openai_api_key=openai_api_key)
    
    # Process or load documentation DB
    if not os.path.exists("stellar_vector_db"):
        logger.info("Processing PDF document to create document DB...")
        doc_db = process_pdf_document(pdf_path)
    else:
        logger.info("Loading existing document database...")
        doc_db = FAISS.load_local("stellar_vector_db", embeddings, allow_dangerous_deserialization=True)
    
    # Process or load conversation DB
    if not os.path.exists("stellar_conversation_db"):
        logger.info("Processing conversation files to create conversation DB...")
        convo_db = create_conversation_db(conversation_dir, batch_size)
    else:
        logger.info("Loading existing conversation database...")
        convo_db = FAISS.load_local("stellar_conversation_db", embeddings, allow_dangerous_deserialization=True)
    
    if not doc_db and not convo_db:
        logger.error("No databases available. Exiting.")
        return None
    
    logger.info("Creating hybrid retriever...")
    doc_retriever = doc_db.as_retriever() if doc_db else None
    convo_retriever = convo_db.as_retriever() if convo_db else None
    hybrid_retriever = HybridRetriever(doc_retriever, convo_retriever)
    
    # Initialize LLM (adjust model parameters as needed)
    llm = ChatOpenAI(
        temperature=0.2,
        openai_api_key=openai_api_key,
        model_name="chatgpt-4o-latest"
    )
    
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=hybrid_retriever,
        return_source_documents=True
    )
    
    logger.info("Hybrid RAG system setup complete.")
    return qa_chain

def query_rag_system(qa_system, query):
    """Query the RAG system and return the answer, with full logging of retrieved text and response."""
    if not qa_system:
        logger.error("RAG system not properly initialized.")
        return "RAG system not properly initialized."
    
    result = qa_system({"query": query})
    answer = result.get("result", "No answer returned.")
    
    # Log the retrieved documents and final answer.
    retrieved_docs = result.get("source_documents", [])
    logger.info("=== RAG Query Debug Info ===")
    logger.info("Query: %s", query)
    logger.info("Retrieved Documents:")
    for i, doc in enumerate(retrieved_docs):
        logger.info("Document %d (first 300 characters): %s", i+1, doc.page_content[:300])
    logger.info("Final Answer: %s", answer)
    
    return answer

###############################################################
#                         DISCORD BOT                         #
###############################################################

# Set up Discord bot with appropriate intents.
intents = discord.Intents.default()
intents.message_content = True
bot = commands.Bot(command_prefix='!', intents=intents)

# Global variable for user conversation history (if you want to store past Q&A)
user_histories = {}

# Initialize the RAG system (hybrid retriever) on bot startup.
qa_system = setup_full_rag_system()

@bot.event
async def on_ready():
    logger.info("Bot connected as %s", bot.user)
    logger.info("Hybrid RAG system is ready!")
    print(f'Bot connected as {bot.user}')
    print("Hybrid RAG system is ready!")

@bot.command(name="scout")
async def scout_command(ctx, *, question=None):
    """Talk to Scout support assistant."""
    user_id = str(ctx.author.id)
    if user_id not in user_histories:
        user_histories[user_id] = []
    
    if not question:
        await ctx.send("Hi there! I'm Scout from the Stellar Support Team. How can I help you today?")
        return

    async with ctx.typing():
        try:
            logger.info("Processing request from %s: '%s'", ctx.author.name, question)
            answer = query_rag_system(qa_system, question)
            
            if not answer.strip():
                answer = "I'm having trouble generating a response right now. Could you rephrase your question?"
                logger.warning("Empty answer detected for query: '%s'", question)
            
            # Update conversation history
            user_histories[user_id].append({"role": "user", "content": question})
            user_histories[user_id].append({"role": "assistant", "content": answer})
            logger.info("Updated conversation history for user %s. Total messages: %d", user_id, len(user_histories[user_id]))
        
        except Exception as e:
            logger.error("Error processing request: %s", str(e))
            logger.error(traceback.format_exc())
            answer = f"An error occurred while processing your request: {str(e)[:200]}... Please try again."
    
    await ctx.send(answer)

@bot.command(name="reset")
async def reset_command(ctx):
    """Reset your conversation with Scout."""
    user_id = str(ctx.author.id)
    user_histories[user_id] = []
    logger.info("Conversation history reset for user %s", user_id)
    await ctx.send("I've reset our conversation. What would you like to discuss next?")

@bot.command(name="debug")
async def debug_command(ctx, *, question=None):
    """Debug mode: shows detailed info about the retrieved documents."""
    if not question:
        await ctx.send("Please provide a question to debug.")
        return
    async with ctx.typing():
        try:
            embeddings = OpenAIEmbeddings(model="text-embedding-ada-002", openai_api_key=openai_api_key)
            db = FAISS.load_local("stellar_vector_db", embeddings, allow_dangerous_deserialization=True)
            docs = db.similarity_search(question, k=2)
            
            debug_info = "**Debug Information**\n\n"
            debug_info += f"**Query:** {question}\n\n"
            debug_info += "**Top 2 Retrieved Documents:**\n"
            for i, doc in enumerate(docs):
                debug_info += f"Doc {i+1}: {doc.page_content[:250]}...\n\n"
            debug_info += f"**Chat History Length:** {len(user_histories.get(str(ctx.author.id), []))}\n"
            logger.info("Debug command invoked by %s for query: '%s'", ctx.author.name, question)
            logger.info("Debug info: %s", debug_info)
        
        except Exception as e:
            debug_info = f"Error during debug: {str(e)}\n{traceback.format_exc()}"
            logger.error("Error in debug command: %s", str(e))
            logger.error(traceback.format_exc())
    
    # Split the message if it's too long
    if len(debug_info) > 1900:
        parts = [debug_info[i:i+1900] for i in range(0, len(debug_info), 1900)]
        for part in parts:
            await ctx.send(part)
    else:
        await ctx.send(debug_info)

bot.run(os.getenv('DISCORD_BOT_TOKEN'))
