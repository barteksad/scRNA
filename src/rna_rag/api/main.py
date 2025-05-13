from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os
import logging
from datetime import datetime
from dotenv import load_dotenv
from rna_rag.rag.pipeline import pipeline

load_dotenv()

# Configure logging
log_file = f'api_log_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="RNA RAG API",
    description="API for RNA-seq data analysis using RAG pipeline",
    version="1.0.0"
)

class QueryRequest(BaseModel):
    question: str
    data: str
    answering_model: str | None = None

class QueryResponse(BaseModel):
    answer: str

@app.post("/query", response_model=QueryResponse)
async def process_query(request: QueryRequest):
    try:
        # Use default model from env if not specified
        answering_model = request.answering_model or os.getenv("DEFAULT_ANSWERING_MODEL")
        logger.info(f"Processing query using model: {answering_model}")
        
        # Process the query using the pipeline
        answer = await pipeline(
            question=request.question,
            data=request.data,
            answering_model=answering_model
        )
        
        return QueryResponse(answer=answer)
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    try:
        # Check if required environment variables are set
        required_vars = [
            "QDRANT_DATABASE_PATH", 
            "QDRANT_COLLECTION_NAME", 
            "QDRANT_URL", 
            "DEFAULT_ANSWERING_MODEL"
        ]
        
        missing_vars = [var for var in required_vars if not os.getenv(var)]
        
        if missing_vars:
            return {
                "status": "warning",
                "message": f"Missing environment variables: {', '.join(missing_vars)}"
            }
            
        # Check if Qdrant configuration exists
        qdrant_path = os.getenv("QDRANT_DATABASE_PATH")
        if not os.path.exists(qdrant_path):
            return {
                "status": "warning", 
                "message": f"Qdrant configuration file not found at {qdrant_path}"
            }
            
        return {"status": "healthy"}
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}", exc_info=True)
        return {"status": "unhealthy", "message": str(e)} 