# backend/main.py
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import os
import uuid
import logging
from dotenv import load_dotenv
from backend.ragengine import RAGEngine

# Load environment
load_dotenv(override=True)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize RAG engine
rag_engine = RAGEngine()

# Create FastAPI app
app = FastAPI(title="Document Q&A System", version="1.0.0")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models
class QueryRequest(BaseModel):
    query: str
    session_id: Optional[str] = None

class QueryResponse(BaseModel):
    answer: str
    sources: List[Dict]
    session_id: str


# Store chat sessions
chat_sessions = {}

@app.get("/")
async def root():
    return {"message": "Document Q&A System API", "status": "running"}

@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    """Ask a question about your documents"""
    
    # Get or create session
    session_id = request.session_id or str(uuid.uuid4())
    
    if session_id not in chat_sessions:
        chat_sessions[session_id] = []
    
    # Generate answer
    try:
        result = rag_engine.generate_answer(
            query=request.query,
            chat_history=chat_sessions[session_id]
        )
    except Exception as e:
        logger.exception("Error generating answer")
        raise HTTPException(
            status_code=500,
            detail=(
                "An internal error occurred while generating the response. "
                "Check server logs for details."
            ),
        )

    # Update chat history
    chat_sessions[session_id].append({"role": "user", "content": request.query})
    chat_sessions[session_id].append({"role": "assistant", "content": result["answer"]})

    # Keep history manageable
    if len(chat_sessions[session_id]) > 20:
        chat_sessions[session_id] = chat_sessions[session_id][-20:]

    return QueryResponse(
        answer=result["answer"],
        sources=result["sources"],
        session_id=session_id
    )

@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    """Upload a document to the knowledge base"""
    
    try:
        content = await file.read()
        url = rag_engine.upload_document(content, file.filename)
        
        return {
            "filename": file.filename,
            "url": url,
            "size": len(content),
            "message": "Document uploaded successfully"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health():
    """Health check endpoint."""
    health_info = {
        "status": "healthy",
        "azure_connected": True,
        "models": {
            "chat": os.getenv("CHAT_MODEL_DEPLOYMENT_NAME"),
            "embedding": os.getenv("EMBEDDING_MODEL_DEPLOYMENT_NAME"),
        },
    }

    return health_info
