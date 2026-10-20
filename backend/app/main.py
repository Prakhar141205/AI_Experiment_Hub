
from fastapi import FastAPI

from app.routers.user import router as user_router
from app.routers.dataset import router as dataset_router
from app.routers.experiment import router as experiment_router
from app.routers.run import router as run_router


app = FastAPI(
    title="AI Experiment Hub",
    description=(
        "A platform to preprocess datasets, train ML models, "
        "and track experiments."
    ),
    version="0.1.0",
)

# Register API routers
app.include_router(user_router)
app.include_router(dataset_router)
app.include_router(experiment_router)
app.include_router(run_router)


@app.get("/health", tags=["Health"])
def health_check() -> dict[str, str]:
    return {
        "status": "healthy",
        "service": "AI Experiment Hub",
    }
