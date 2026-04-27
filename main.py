import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import PyPDF2
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Initialize the FastAPI application
app = FastAPI(
    title="Enterprise AI Orchestrator",
    description="Python microservice for processing documents via LangChain/RAG",
    version="1.0.0"
)

# Define the exact JSON payload we expect from the network
class DocumentEvent(BaseModel):
    document_id: str
    filename: str
    file_path: str

# 👉 RESTORED: The mandatory health check for DevOps/Infrastructure monitoring
@app.get("/health")
async def health_check():
    """Simple endpoint to verify the AI engine is online."""
    return {"status": "online", "service": "ai-orchestrator"}

@app.post("/process-document")
async def process_document(event: DocumentEvent):
    print(f"\n📥 [AI ENGINE] Received task to process: {event.document_id}")
    
    # 1. SECURITY CHECK: Verify the file actually exists where the BFF claims it is
    if not os.path.exists(event.file_path):
        print(f"❌ [AI ENGINE ERROR] File not found at {event.file_path}")
        raise HTTPException(status_code=404, detail=f"File not found at {event.file_path}")

    try:
        # 2. EXTRACTION: Open the binary PDF and rip the text out of it
        print(f"📄 [AI ENGINE] Extracting text from: {event.filename}...")
        text_content = ""
        
        with open(event.file_path, "rb") as file:
            reader = PyPDF2.PdfReader(file)
            for page in reader.pages:
                extracted = page.extract_text()
                if extracted:
                    text_content += extracted + "\n"
        
        print(f"✅ [AI ENGINE] Success! Extracted {len(text_content)} characters.")

        # 3. CHUNKING: Split the massive text into smaller, overlapping chunks.
        # We use overlap so we don't accidentally cut a sentence or concept in half.
        print("✂️ [AI ENGINE] Chunking text for Vector Database ingestion...")
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, # 1000 characters per chunk
            chunk_overlap=200, # 200 characters of overlap between chunks
            length_function=len,
        )
        
        chunks = text_splitter.split_text(text_content)
        print(f"✅ [AI ENGINE] Created {len(chunks)} overlapping chunks.")

        # 4. (Future Step) Convert chunks to Vector Embeddings and store in Pinecone/ChromaDB
        
        return {
            "status": "success", 
            "document_id": event.document_id,
            "chunks_created": len(chunks),
            "message": f"Successfully ripped and chunked {event.filename}."
        }

    except Exception as e:
        print(f"❌ [AI ENGINE ERROR] {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))