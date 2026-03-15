# backend/main.py
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import os
import uuid
import logging
import requests
from dotenv import load_dotenv
from azure.identity import DefaultAzureCredential
from azure.ai.projects import AIProjectClient
from backend.ragengine import RAGEngine

# Load environment
load_dotenv(override=True)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Azure client
try:
    project_client = AIProjectClient(
        endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        credential=DefaultAzureCredential()
    )
    logger.info("✅ Connected to Azure AI Foundry")
except Exception as e:
    logger.error(f"❌ Failed to connect: {e}")
    raise

# Initialize RAG engine
rag_engine = RAGEngine(project_client)

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

class DocumentResponse(BaseModel):
    filename: str
    url: str
    size: int

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

@app.get("/debug/deployments")
async def debug_deployments():
    """Debug endpoint to list available OpenAI deployments."""
    deployments = _get_openai_deployments()
    return {"deployments": deployments or []}


def _get_openai_deployments() -> Optional[List[str]]:
    """Fetch list of OpenAI deployments."""
    endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    api_version = os.getenv("AZURE_OPENAI_API_VERSION")
    api_key = os.getenv("AZURE_OPENAI_KEY") or os.getenv("AZURE_SEARCH_KEY")

    if not all([endpoint, api_version, api_key]):
        return None

    url = f"{endpoint.rstrip('/')}/deployments?api-version={api_version}"
    try:
        resp = requests.get(url, headers={"api-key": api_key}, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        return [d.get("name") for d in data.get("data", []) if isinstance(d, dict)]
    except Exception:
        return None


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

    if os.getenv("ENABLE_DEBUG_ENDPOINTS") == "1":
        health_info["openai_deployments"] = _get_openai_deployments()

    return health_info