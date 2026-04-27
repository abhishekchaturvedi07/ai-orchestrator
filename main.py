from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# Initialize the FastAPI application
app = FastAPI(
    title="Enterprise AI Orchestrator",
    description="Python microservice for processing documents via LangChain/RAG",
    version="1.0.0"
)

# Define the data shape we expect to receive (Pydantic makes this strictly typed)
class DocumentEvent(BaseModel):
    document_id: str
    filename: str
    file_path: str

@app.get("/health")
async def health_check():
    """Simple endpoint to verify the AI engine is online."""
    return {"status": "online", "service": "ai-orchestrator"}

@app.post("/process-document")
async def process_document(event: DocumentEvent):
    """
    This endpoint will eventually be called by our Node.js BFF (or a Kafka queue).
    It receives the path to the securely stored PDF and will start the AI chunking.
    """
    print(f"📥 [AI ENGINE] Received task to process: {event.document_id}")
    print(f"🔍 [AI ENGINE] Locating file at: {event.file_path}")
    
    # TODO: Initialize LangChain, load the PDF, chunk the text, and generate embeddings.
    
    return {
        "status": "success", 
        "message": f"Successfully queued {event.filename} for AI processing."
    }