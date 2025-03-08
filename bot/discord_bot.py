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

from pydantic import Field
from typing import Any, List

# For custom prompt templates and chains
from langchain import PromptTemplate
from langchain.chains.llm import LLMChain

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

openai_api_key = os.getenv("OPENAI_API_KEY")
pdf_path = "Stellar.pdf"          # Path to your PDF file
conversation_dir = "data"         # Directory containing conversation JSON files
admin_transcripts_dir = "admin_transcripts"  # Directory containing admin transcript text files
batch_size = 100                  # Process files in batches

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
    db = FAISS.from_documents(
        documents,
        OpenAIEmbeddings(model="text-embedding-ada-002", openai_api_key=openai_api_key)
    )
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
#                     ADMIN TRANSCRIPTS PROCESSING            #
###############################################################

def process_admin_transcript_file(file_path):
    """Process a single admin transcript text file."""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            transcript = f.read()
        if len(transcript) < 100:
            return None
        cleaned_text = clean_text(transcript)
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
                    "source": os.path.basename(file_path),
                    "source_type": "admin_transcript"
                }
            )
            for chunk in chunks
        ]
        return documents
    except Exception as e:
        logger.error("Error processing admin transcript file %s: %s", file_path, str(e))
        return None

def batch_process_admin_transcripts(directory, batch_size=100):
    """Process admin transcript files in batches."""
    file_paths = glob.glob(os.path.join(directory, "**/*.txt"), recursive=True)
    logger.info("Found %d admin transcript files", len(file_paths))
    all_documents = []
    total_batches = (len(file_paths) + batch_size - 1) // batch_size
    for batch_idx in range(total_batches):
        start_idx = batch_idx * batch_size
        end_idx = min((batch_idx + 1) * batch_size, len(file_paths))
        batch_files = file_paths[start_idx:end_idx]
        logger.info("Processing admin transcript batch %d/%d (%d files)", batch_idx + 1, total_batches, len(batch_files))
        for file_path in tqdm(batch_files):
            docs = process_admin_transcript_file(file_path)
            if docs:
                all_documents.extend(docs)
    logger.info("Total admin transcripts processed: %d", len(all_documents))
    return all_documents

def create_admin_transcripts_db(admin_transcripts_dir, batch_size):
    """Create a vector database from all admin transcript files."""
    documents = batch_process_admin_transcripts(admin_transcripts_dir, batch_size)
    if not documents:
        logger.warning("No valid admin transcripts found.")
        return None
    logger.info("Creating vector database from admin transcripts...")
    embeddings = OpenAIEmbeddings(model="text-embedding-ada-002", openai_api_key=openai_api_key)
    admin_db = FAISS.from_documents(documents, embeddings)
    admin_db.save_local("stellar_admin_transcripts_db")
    logger.info("Created admin transcripts database with %d entries", len(documents))
    return admin_db

###############################################################
#                     HYBRID RETRIEVER SETUP                  #
###############################################################

class HybridRetrieverExtended:
    """
    A simple hybrid retriever that queries three separate retrievers and concatenates results.
    """
    def __init__(self, doc_retriever, convo_retriever, admin_retriever, k: int = 5,
                 doc_weight: float = 0.5, convo_weight: float = 0.3, admin_weight: float = 0.2):
        self.doc_retriever = doc_retriever
        self.convo_retriever = convo_retriever
        self.admin_retriever = admin_retriever
        self.doc_weight = doc_weight
        self.convo_weight = convo_weight
        self.admin_weight = admin_weight
        self.k = k

    def get_relevant_documents(self, query: str) -> List[Any]:
        doc_k = max(1, int(self.k * self.doc_weight))
        convo_k = max(1, int(self.k * self.convo_weight))
        admin_k = max(1, self.k - doc_k - convo_k)
        doc_results = self.doc_retriever.get_relevant_documents(query, k=doc_k) if self.doc_retriever else []
        convo_results = self.convo_retriever.get_relevant_documents(query, k=convo_k) if self.convo_retriever else []
        admin_results = self.admin_retriever.get_relevant_documents(query, k=admin_k) if self.admin_retriever else []
        return doc_results + convo_results + admin_results

###############################################################
#                     FULL SYSTEM SETUP & QUERY               #
###############################################################

def setup_full_rag_system():
    """Process documents, conversations, and admin transcripts, and set up a custom chain that incorporates conversation history."""
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

    # Process or load admin transcripts DB
    if not os.path.exists("stellar_admin_transcripts_db"):
        logger.info("Processing admin transcript files to create admin transcripts DB...")
        admin_db = create_admin_transcripts_db(admin_transcripts_dir, batch_size)
    else:
        logger.info("Loading existing admin transcripts database...")
        admin_db = FAISS.load_local("stellar_admin_transcripts_db", embeddings, allow_dangerous_deserialization=True)

    if not doc_db and not convo_db and not admin_db:
        logger.error("No databases available. Exiting.")
        return None

    logger.info("Creating extended hybrid retriever...")
    doc_retriever = doc_db.as_retriever() if doc_db else None
    convo_retriever = convo_db.as_retriever() if convo_db else None
    admin_retriever = admin_db.as_retriever() if admin_db else None

    hybrid_retriever = HybridRetrieverExtended(
        doc_retriever=doc_retriever,
        convo_retriever=convo_retriever,
        admin_retriever=admin_retriever
    )

    # Initialize LLM
    llm = ChatOpenAI(
        temperature=0.2,
        openai_api_key=openai_api_key,
        model_name="chatgpt-4o-latest"
    )

    # Create a custom prompt template that includes conversation history.
    template = """
You are Scout, a friendly and knowledgeable member of the Stellar Support Team on Discord.
Your role is to help users with queries related to Stellar's products, services, and troubleshooting.
Here is the conversation history:
{history}

Context:
{context}

Question:
{query}

Answer:"""
    prompt_template = PromptTemplate(
        input_variables=["context", "history", "query"],
        template=template
    )

    # Create an LLMChain with our custom prompt.
    llm_chain = LLMChain(llm=llm, prompt=prompt_template)

    # Instead of using RetrievalQA, we build our own query function.
    def qa_system(query: str, conversation_history: str = ""):
        # Retrieve relevant documents from the hybrid retriever.
        docs = hybrid_retriever.get_relevant_documents(query)
        context = "\n".join([doc.page_content for doc in docs])
        answer = llm_chain.predict(context=context, query=query, history=conversation_history)
        return answer, docs

    logger.info("Extended hybrid system setup complete.")
    return qa_system

def query_rag_system(qa_system, query, conversation_history=""):
    """Query the system and return the answer along with logging debug info."""
    if not qa_system:
        logger.error("System not properly initialized.")
        return "System not properly initialized."

    answer, retrieved_docs = qa_system(query, conversation_history)
    logger.info("=== Query Debug Info ===")
    logger.info("Query: %s", query)
    for i, doc in enumerate(retrieved_docs):
        logger.info("Document %d (first 300 characters): %s", i + 1, doc.page_content[:300])
    logger.info("Final Answer: %s", answer)
    return answer

###############################################################
#                         DISCORD BOT                         #
###############################################################

# Define a staff channel ID for notifications (update with your actual channel ID)
STAFF_CHANNEL_ID = int(os.getenv("STAFF_CHANNEL_ID", "123456789012345678"))

# Create a view with a "Call Human Assistance" button.
class CallHumanAssistanceView(discord.ui.View):
    def __init__(self, *, timeout: float = None):
        super().__init__(timeout=timeout)

    @discord.ui.button(
        label="Call Human Assistance",
        style=discord.ButtonStyle.primary,
        custom_id="tickettool_call_human_assistance"  # Customize for TicketTool integration if needed
    )
    async def call_human(self, interaction: discord.Interaction, button: discord.ui.Button):
        # Notify the staff channel that a user has requested assistance.
        staff_channel = bot.get_channel(STAFF_CHANNEL_ID)
        if staff_channel:
            await staff_channel.send(f"User {interaction.user.mention} has requested human assistance!")
        else:
            logger.warning("Staff channel not found. Please check your STAFF_CHANNEL_ID.")
        # Acknowledge the interaction privately.
        await interaction.response.send_message(
            "Your request for human assistance has been sent to our staff.",
            ephemeral=True
        )

intents = discord.Intents.default()
intents.message_content = True
bot = commands.Bot(command_prefix='!', intents=intents)

user_histories = {}

# Initialize the custom RAG system on bot startup.
qa_system = setup_full_rag_system()

@bot.event
async def on_ready():
    logger.info("Bot connected as %s", bot.user)
    logger.info("Extended hybrid system is ready!")
    print(f'Bot connected as {bot.user}')
    print("Extended hybrid system is ready!")

@bot.command(name="scout")
async def scout_command(ctx, *, question=None):
    """Talk to Scout support assistant."""
    user_id = str(ctx.author.id)
    if user_id not in user_histories:
        user_histories[user_id] = []

    if not question:
        await ctx.send("Hi there! I'm Scout from the Stellar Support Team. How can I help you today?")
        return

    # Build conversation history from previous messages.
    conversation_history = "\n".join(
        f"{msg['role'].upper()}: {msg['content']}" for msg in user_histories[user_id]
    )

    async with ctx.typing():
        try:
            logger.info("Processing request from %s: '%s'", ctx.author.name, question)
            answer = query_rag_system(qa_system, question, conversation_history)
            if not answer.strip():
                answer = "I'm having trouble generating a response right now. Could you rephrase your question?"
                logger.warning("Empty answer detected for query: '%s'", question)

            # Record both the user's question and the assistant's answer.
            user_histories[user_id].append({"role": "user", "content": question})
            user_histories[user_id].append({"role": "assistant", "content": answer})
            logger.info("Updated conversation history for user %s. Total messages: %d", user_id, len(user_histories[user_id]))
        except Exception as e:
            logger.error("Error processing request: %s", str(e))
            logger.error(traceback.format_exc())
            answer = f"An error occurred while processing your request: {str(e)[:200]}... Please try again."

    await ctx.send(answer)
    view = CallHumanAssistanceView(timeout=None)
    await ctx.send("If you need human assistance, please click the button below:", view=view)

@bot.command(name="reset")
async def reset_command(ctx):
    """Reset your conversation with Scout."""
    user_id = str(ctx.author.id)
    user_histories[user_id] = []
    logger.info("Conversation history reset for user %s", user_id)
    await ctx.send("I've reset our conversation. What would you like to discuss next?")

@bot.command(name="debug")
async def debug_command(ctx, *, question=None):
    """Debug mode: shows detailed info about the retrieved documents from all databases."""
    if not question:
        await ctx.send("Please provide a question to debug.")
        return
    async with ctx.typing():
        try:
            retrieved_docs = qa_system.__globals__['hybrid_retriever'].get_relevant_documents(question)
            debug_info = "**Debug Information**\n\n"
            debug_info += f"**Query:** {question}\n\n"
            debug_info += "**Retrieved Documents (from all databases):**\n"
            for i, doc in enumerate(retrieved_docs):
                source = doc.metadata.get("source_type", "unknown")
                debug_info += f"Doc {i+1} (Source: {source}): {doc.page_content[:250]}...\n\n"
            debug_info += f"**Chat History Length:** {len(user_histories.get(str(ctx.author.id), []))}\n"
            logger.info("Debug command invoked by %s for query: '%s'", ctx.author.name, question)
            logger.info("Debug info: %s", debug_info)
        except Exception as e:
            debug_info = f"Error during debug: {str(e)}\n{traceback.format_exc()}"
            logger.error("Error in debug command: %s", str(e))
            logger.error(traceback.format_exc())

    if len(debug_info) > 1900:
        parts = [debug_info[i:i+1900] for i in range(0, len(debug_info), 1900)]
        for part in parts:
            await ctx.send(part)
    else:
        await ctx.send(debug_info)

@bot.command(name="assist")
async def assist_command(ctx):
    """Sends a message with a button to call human assistance."""
    view = CallHumanAssistanceView(timeout=None)
    await ctx.send("", view=view)

bot.run(os.getenv('DISCORD_BOT_TOKEN'))
