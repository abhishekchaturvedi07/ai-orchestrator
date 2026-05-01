import os
import threading 
import jwt #[cite: 3]
from fastapi import FastAPI, HTTPException, Depends #[cite: 1, 3]
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials #[cite: 3]
from pydantic import BaseModel
import PyPDF2
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Import OpenAI components (We will route these to LM Studio)
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

from worker import start_worker
from prometheus_fastapi_instrumentator import Instrumentator

# Import our Agentic State Machine (Phase 11)
from agent import build_agentic_graph

# Initialize the FastAPI application
app = FastAPI(
    title="Enterprise AI Orchestrator",
    description="Python microservice for processing documents via LangChain/RAG",
    version="1.0.0"
)

# --- SECURITY CONFIGURATION (Layer 0) ---
security = HTTPBearer() #[cite: 3]
JWT_SECRET = "enterprise-secret-2026" # Match this with your BFF/Identity secret #[cite: 3]

# --- START UP THE AI BRAIN (VIA LM STUDIO) ---
print("🧠 [AI ENGINE] Connecting to LM Studio Local Server...")

# Instrument the app to expose a /metrics endpoint for Prometheus
Instrumentator().instrument(app).expose(app) #[cite: 1]

# THE DROP-IN PATTERN: We use the official OpenAI class, but hijack the URL.
embeddings = OpenAIEmbeddings(
    openai_api_base="http://host.docker.internal:1234/v1",
    openai_api_key="lm-studio",
    model="text-embedding-nomic-embed-text-v1.5", # 👉 Explicitly name your LM Studio model
    check_embedding_ctx_length=False # 👉 Forces LangChain to send raw strings, NOT integers
) #[cite: 1]

print("💾 [AI ENGINE] Connecting to Chroma Vector Database...")
# This will create a local folder called 'chroma_db' to permanently save our vectors
vector_store = Chroma(
    collection_name="enterprise_documents",
    embedding_function=embeddings,
    persist_directory="./chroma_db"
) #[cite: 1]

# 👉 Compile the Agentic Graph once on startup
print("⚙️ [AI ENGINE] Compiling Agentic Graph...")
agent_workflow = build_agentic_graph(vector_store) #[cite: 1]

print("🚀 [AI ENGINE] System Ready!")
# ------------------------------

class DocumentEvent(BaseModel):
    document_id: str
    filename: str
    file_path: str #[cite: 1]

class AskEvent(BaseModel):
    document_id: str
    question: str #[cite: 1, 3]

@app.get("/health")
async def health_check():
    return {"status": "online", "service": "ai-orchestrator"} #[cite: 1]

"""
ARCHITECTURAL NOTE [LEGACY API]:
This /process-document endpoint is our Phase 1 Synchronous REST implementation. 
"""
@app.post("/process-document")
async def process_document(event: DocumentEvent):
    print(f"\n📥 [AI ENGINE] Received task to process: {event.document_id}") #[cite: 1]
    
    if not os.path.exists(event.file_path):
        print(f"❌ [AI ENGINE ERROR] File not found at {event.file_path}") #[cite: 1]
        raise HTTPException(status_code=404, detail=f"File not found at {event.file_path}") #[cite: 1]

    try:
        # 1. EXTRACTION
        print(f"📄 [AI ENGINE] Extracting text from: {event.filename}...") #[cite: 1]
        text_content = ""
        with open(event.file_path, "rb") as file:
            reader = PyPDF2.PdfReader(file)
            for page in reader.pages:
                extracted = page.extract_text()
                if extracted:
                    text_content += extracted + "\n" #[cite: 1]
        
        # 2. CHUNKING
        print("✂️ [AI ENGINE] Chunking text...") #[cite: 1]
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, 
            chunk_overlap=200,
            length_function=len,
        )
        raw_chunks = text_splitter.split_text(text_content)
        
        # 👉 Strip out any empty chunks that might crash LM Studio
        chunks = [chunk for chunk in raw_chunks if chunk.strip()] #[cite: 1]
        
        print(f"✅ [AI ENGINE] Created {len(chunks)} valid overlapping chunks.") #[cite: 1]

        # 3. VECTOR INGESTION
        print("🧮 [AI ENGINE] Requesting Embeddings from LM Studio and saving to ChromaDB...") #[cite: 1]
        
        documents = [
            Document(
                page_content=chunk, 
                metadata={"source": event.filename, "document_id": event.document_id}
            )
            for chunk in chunks
        ]
        
        vector_store.add_documents(documents) #[cite: 1]
        print("✅ [AI ENGINE] Vectors successfully saved to Disk!") #[cite: 1]

        return {
            "status": "success", 
            "document_id": event.document_id,
            "chunks_created": len(chunks),
            "message": f"Successfully chunked and embedded {event.filename} via LM Studio."
        } #[cite: 1]

    except Exception as e:
        print(f"❌ [AI ENGINE ERROR] {str(e)}") #[cite: 1]
        raise HTTPException(status_code=500, detail=str(e)) #[cite: 1]


# --- THE SECURED CHAT ENGINE ---

@app.post("/ask-copilot")
async def ask_copilot(event: AskEvent, credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Validates user identity before allowing any AI interaction.""" #[cite: 3]
    
    # 1. IDENTITY PERIMETER CHECK (Layer 0)
    try:
        token = credentials.credentials #[cite: 3]
        payload = jwt.decode(token, JWT_SECRET, algorithms=["HS256"]) #[cite: 3]
        # user_id = payload.get("sub") # Extract the unique user identifier #[cite: 3]
        user_id = payload.get("id")
        print(f"👤 [AUTH] Request validated for User: {user_id}") #[cite: 3]
    except Exception:
        print("🚨 [AUTH ERROR] Invalid or missing token!") #[cite: 3]
        raise HTTPException(status_code=401, detail="Unauthorized: Invalid Identity Token") #[cite: 3]

    print(f"\n💬 [AI ENGINE] Question received for doc {event.document_id}: '{event.question}'") #[cite: 3]

    # 👉 THE FEATURE FLAG
    AI_MODE = os.getenv("AI_MODE", "STANDARD").upper() #[cite: 3]
    print(f"🎛️ [FEATURE FLAG] Operating Mode: {AI_MODE}") #[cite: 3]

    try:
        if AI_MODE == "AGENTIC":
            # --- THE SECURED LANGGRAPH PATH ---
            print("🧠 [AI ENGINE] Handing request to LangGraph Semantic Router...") #[cite: 3]
            
            # Run the state machine with injected User Identity
            result = agent_workflow.invoke({
                "question": event.question,
                "document_id": event.document_id,
                "user_id": user_id, # 👉 Injecting user_id for RLS/Security nodes
                "context": "",
                "route": "",
                "answer": ""
            }) #[cite: 3]
            
            # Dynamic Source Attribution
            if result["route"] == "DATABASE":
                sources = [event.document_id]
            elif result["route"] == "WEB":
                sources = ["Live Internet Search via DuckDuckGo"]
            else:
                sources = ["General AI Knowledge"]
                
            return {
                "answer": result["answer"],
                "sources": sources
            } #[cite: 3]

        else:
            # --- THE LEGACY STANDARD PATH ---
            print("🔍 [AI ENGINE] Executing Standard RAG Pipeline...") #[cite: 3]
            
            results = vector_store.similarity_search(
                query=event.question,
                k=3,
                filter={"document_id": event.document_id} 
            ) #[cite: 3]

            if not results:
                return {
                    "answer": "I couldn't find any relevant information in this document.", 
                    "sources": []
                } #[cite: 3]

            context = "\n\n".join([doc.page_content for doc in results])
            
            llm = ChatOpenAI(
                openai_api_base="http://host.docker.internal:1234/v1",
                openai_api_key="lm-studio",
                temperature=0.1 
            ) #[cite: 3]

            prompt_template = ChatPromptTemplate.from_messages([
                ("system", "You are an expert enterprise AI assistant. Answer based on CONTEXT:\n{context}"),
                ("user", "{question}")
            ]) #[cite: 3]

            chain = prompt_template | llm
            response = chain.invoke({"context": context, "question": event.question}) #[cite: 3]

            return {
                "answer": response.content,
                "sources": [event.document_id]
            } #[cite: 3]

    except Exception as e:
        print(f"❌ [AI ENGINE ERROR] {str(e)}") #[cite: 3]
        raise HTTPException(status_code=500, detail=str(e)) #[cite: 3]


# --- ⚡ ASYNCHRONOUS WORKER ---
@app.on_event("startup")
def startup_event():
    print("🚀 [EVENT MESH] Spinning up Background RabbitMQ Consumer Thread...") #[cite: 3]
    worker_thread = threading.Thread(target=start_worker, daemon=True) #[cite: 3]
    worker_thread.start() #[cite: 3]