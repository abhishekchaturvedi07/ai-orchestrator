import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import PyPDF2
from langchain_text_splitters import RecursiveCharacterTextSplitter

# 👉 NEW: Import OpenAI components (We will route these to LM Studio)
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

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



# --- THE CHAT ENGINE ---

# Define the shape of the incoming question
class AskEvent(BaseModel):
    document_id: str
    question: str

@app.post("/ask-copilot")
async def ask_copilot(event: AskEvent):
    print(f"\n💬 [AI ENGINE] Question received for doc {event.document_id}: '{event.question}'")

    try:
        # 1. RETRIEVAL: Find the top 3 most relevant chunks in ChromaDB
        print("🔍 [AI ENGINE] Searching Vector Database for the answer...")
        
        # We explicitly filter by document_id so it doesn't accidentally pull answers from a different PDF!
        results = vector_store.similarity_search(
            query=event.question,
            k=3,
            filter={"document_id": event.document_id} 
        )

        if not results:
            return {
                "answer": "I couldn't find any relevant information in this document.", 
                "sources": []
            }

        # Combine the 3 chunks into a single giant string of context
        context = "\n\n".join([doc.page_content for doc in results])
        print(f"✅ [AI ENGINE] Found {len(results)} relevant chunks.")

        # 2. GENERATION: Connect to your Generative LLM (Llama 3 in LM Studio)
        print("🧠 [AI ENGINE] Sending context to Generative LLM for final answer...")
        
        # We set temperature to 0.1 so the AI doesn't hallucinate. It stays strictly factual.
        llm = ChatOpenAI(
            openai_api_base="http://localhost:1234/v1",
            openai_api_key="lm-studio",
            temperature=0.1 
        )

        # 3. PROMPT ENGINEERING: The Enterprise Guardrails
        prompt_template = ChatPromptTemplate.from_messages([
            ("system", "You are an expert enterprise AI assistant. Answer the user's question using ONLY the provided CONTEXT. If the answer is not in the context, say 'I don't know based on the provided document.' Do not make things up.\n\nCONTEXT:\n{context}"),
            ("user", "{question}")
        ])

        # 4. CHAIN IT TOGETHER AND RUN IT
        chain = prompt_template | llm
        response = chain.invoke({"context": context, "question": event.question})

        print("✅ [AI ENGINE] Answer generated successfully!")

        return {
            "answer": response.content,
            "sources": [event.document_id]
        }

    except Exception as e:
        print(f"❌ [AI ENGINE ERROR] {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))