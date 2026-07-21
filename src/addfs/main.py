from fastapi import FastAPI
from addfs.api.upload import router as upload_router
app = FastAPI(
    title="ADDFS",
    description="Automated Deepfake Detection System",
    version="0.1.0",
)
app.include_router(upload_router)

@app.get("/")
def read_root() -> dict[str, str]:
    return {
        "application": "ADDFS",
        "status": "running",
        "message": "Automated Deepfake Detection System API is operational.",
    }


@app.get("/health")
def health_check() -> dict[str, str]:
    return {"status": "healthy"}