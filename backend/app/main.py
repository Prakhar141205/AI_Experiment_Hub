
from fastapi import FastAPI

from app.routers.user import router as user_router

app = FastAPI(
    title="AI Experiment Hub",
    description="A platform to preprocess datasets, train ML models, and track experiments.",
    version="0.1.0",
)

app.include_router(user_router)


@app.get("/health", tags=["Health"])
def health_check():
    return {
        "status": "healthy",
        "service": "AI Experiment Hub",
    }
