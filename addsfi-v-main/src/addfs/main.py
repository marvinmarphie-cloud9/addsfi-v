from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from addfs.api.upload import router as upload_router

app = FastAPI(
    title="ADDFS",
    description="Automated Deepfake Detection System",
    version="0.1.0",
)
app.include_router(upload_router)

STATIC_DIR = Path(__file__).parent / "static"

# Serve the CSS/JS-in-HTML bundle at /static/*
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/")
def serve_ui() -> FileResponse:
    """Serve the web interface at the root URL."""
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/status")
def read_root() -> dict[str, str]:
    return {
        "application": "ADDFS",
        "status": "running",
        "message": "Automated Deepfake Detection System API is operational.",
    }


@app.get("/health")
def health_check() -> dict[str, str]:
    return {"status": "healthy"}
