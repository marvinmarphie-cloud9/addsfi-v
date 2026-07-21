from pathlib import Path
from uuid import uuid4

import cv2
from fastapi import APIRouter, File, HTTPException, UploadFile

router = APIRouter(
    prefix="/upload",
    tags=["Video Upload"],
)

UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

ALLOWED_EXTENSIONS = {".mp4", ".mkv", ".avi", ".mov"}
MAX_FILE_SIZE = 100 * 1024 * 1024  # 100 MB


@router.post("/")
async def upload_video(file: UploadFile = File(...)) -> dict:
    """Validate a video, save it safely, and return its metadata."""

    original_filename = file.filename or "unnamed-video"
    extension = Path(original_filename).suffix.lower()

    if extension not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail="Unsupported file type. Use MP4, MKV, AVI, or MOV.",
        )

    file_bytes = await file.read()

    if not file_bytes:
        raise HTTPException(
            status_code=400,
            detail="The uploaded file is empty.",
        )

    if len(file_bytes) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=413,
            detail="The uploaded file exceeds the 100 MB limit.",
        )

    safe_filename = f"{uuid4().hex}{extension}"
    saved_path = UPLOAD_DIR / safe_filename
    saved_path.write_bytes(file_bytes)

    video = cv2.VideoCapture(str(saved_path))

    if not video.isOpened():
        saved_path.unlink(missing_ok=True)
        raise HTTPException(
            status_code=400,
            detail="The uploaded file could not be opened as a valid video.",
        )

    fps = float(video.get(cv2.CAP_PROP_FPS))
    frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))

    video.release()

    if frame_count <= 0 or width <= 0 or height <= 0:
        saved_path.unlink(missing_ok=True)
        raise HTTPException(
            status_code=400,
            detail="The video does not contain readable frames.",
        )

    duration_seconds = frame_count / fps if fps > 0 else 0.0

    return {
        "status": "uploaded",
        "original_filename": original_filename,
        "stored_filename": safe_filename,
        "size_bytes": len(file_bytes),
        "resolution": {
            "width": width,
            "height": height,
        },
        "fps": round(fps, 2),
        "frame_count": frame_count,
        "duration_seconds": round(duration_seconds, 2),
    }