import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import PyPDF2
from langchain_text_splitters import RecursiveCharacterTextSplitter

# 👉 NEW: Import OpenAI components (We will route these to LM Studio)
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

# Initialize the FastAPI application
app = FastAPI(
    title="Enterprise AI Orchestrator",
    description="Python microservice for processing documents via LangChain/RAG",
    version="1.0.0"
)

# --- START UP THE AI BRAIN (VIA LM STUDIO) ---
print("🧠 [AI ENGINE] Connecting to LM Studio Local Server...")

# THE DROP-IN PATTERN: We use the official OpenAI class, but hijack the URL.
embeddings = OpenAIEmbeddings(
    openai_api_base="http://localhost:1234/v1",
    openai_api_key="lm-studio",
    model="text-embedding-nomic-embed-text-v1.5", # 👉 FIX 1: Explicitly name your LM Studio model
    check_embedding_ctx_length=False # 👉 FIX 2: Forces LangChain to send raw strings, NOT integers
)

print("💾 [AI ENGINE] Connecting to Chroma Vector Database...")
# This will create a local folder called 'chroma_db' to permanently save our vectors
vector_store = Chroma(
    collection_name="enterprise_documents",
    embedding_function=embeddings,
    persist_directory="./chroma_db"
)
print("🚀 [AI ENGINE] System Ready!")
# ------------------------------

class DocumentEvent(BaseModel):
    document_id: str
    filename: str
    file_path: str

@app.get("/health")
async def health_check():
    return {"status": "online", "service": "ai-orchestrator"}

@app.post("/process-document")
async def process_document(event: DocumentEvent):
    print(f"\n📥 [AI ENGINE] Received task to process: {event.document_id}")
    
    if not os.path.exists(event.file_path):
        print(f"❌ [AI ENGINE ERROR] File not found at {event.file_path}")
        raise HTTPException(status_code=404, detail=f"File not found at {event.file_path}")

    try:
        # 1. EXTRACTION
        print(f"📄 [AI ENGINE] Extracting text from: {event.filename}...")
        text_content = ""
        with open(event.file_path, "rb") as file:
            reader = PyPDF2.PdfReader(file)
            for page in reader.pages:
                extracted = page.extract_text()
                if extracted:
                    text_content += extracted + "\n"
        
        # 2. CHUNKING
        print("✂️ [AI ENGINE] Chunking text...")
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, 
            chunk_overlap=200,
            length_function=len,
        )
        raw_chunks = text_splitter.split_text(text_content)
        
        # 👉 FIX 3: Strip out any empty chunks that might crash LM Studio
        chunks = [chunk for chunk in raw_chunks if chunk.strip()]
        
        print(f"✅ [AI ENGINE] Created {len(chunks)} valid overlapping chunks.")

        # 3. VECTOR INGESTION (Now powered by Nomic in LM Studio!)
        print("🧮 [AI ENGINE] Requesting Embeddings from LM Studio and saving to ChromaDB...")
        
        documents = [
            Document(
                page_content=chunk, 
                metadata={"source": event.filename, "document_id": event.document_id}
            )
            for chunk in chunks
        ]
        
        # Fire them into the database! (This triggers the API call to LM Studio)
        vector_store.add_documents(documents)
        print("✅ [AI ENGINE] Vectors successfully saved to Disk!")

        return {
            "status": "success", 
            "document_id": event.document_id,
            "chunks_created": len(chunks),
            "message": f"Successfully chunked and embedded {event.filename} via LM Studio."
        }

    except Exception as e:
        print(f"❌ [AI ENGINE ERROR] {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))