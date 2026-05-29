import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from dotenv import load_dotenv
import os
import logging

load_dotenv()

from config.settings import get_settings
from api.routes.analyze import router as analyze_router
from api.routes.chat import router as chat_router
from api.routes.health import router as health_router

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting Social Media RAG Chatbot Backend...")
    logger.info(f"ChromaDB persist dir: {settings.CHROMA_PERSIST_DIR}")
    logger.info(f"Embedding model: {settings.EMBEDDING_MODEL}")
    os.makedirs(settings.CHROMA_PERSIST_DIR, exist_ok=True)
    os.makedirs("./temp_downloads", exist_ok=True)
    yield
    logger.info("Shutting down...")


app = FastAPI(
    title="Social Media RAG Chatbot",
    description="RAG-powered chatbot for comparing social media videos",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[settings.FRONTEND_URL, "http://localhost:5173", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health_router, prefix="/api", tags=["Health"])
app.include_router(analyze_router, prefix="/api", tags=["Analysis"])
app.include_router(chat_router, prefix="/api", tags=["Chat"])

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
    )