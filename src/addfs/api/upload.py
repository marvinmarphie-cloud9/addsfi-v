from pathlib import Path
from uuid import uuid4

import cv2
from fastapi import APIRouter, File, HTTPException, UploadFile

from addfs.detection.deepfake_detector import DeepfakeDetector
from addfs.preprocessing.face_cropper import crop_faces
from addfs.preprocessing.face_detector import FaceDetector
from addfs.preprocessing.frame_extractor import extract_frames


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

    try:
        fps = float(video.get(cv2.CAP_PROP_FPS))
        frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    finally:
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


@router.post("/extract-frames/{stored_filename}")
def extract_uploaded_video_frames(stored_filename: str) -> dict:
    """Extract sampled frames from a previously uploaded video."""

    video_path = UPLOAD_DIR / stored_filename

    if not video_path.exists():
        raise HTTPException(
            status_code=404,
            detail="Uploaded video was not found.",
        )

    try:
        frame_paths = extract_frames(
            video_path=video_path,
            sample_every_n_frames=30,
            max_frames=50,
        )
    except (FileNotFoundError, ValueError) as error:
        raise HTTPException(
            status_code=400,
            detail=str(error),
        ) from error

    return {
        "status": "frames_extracted",
        "stored_filename": stored_filename,
        "frame_count": len(frame_paths),
        "frames": [str(path) for path in frame_paths],
    }


@router.post("/detect-faces/{stored_filename}")
def detect_faces_in_uploaded_video(stored_filename: str) -> dict:
    """
    Extract frames, detect faces, crop them, and run temporary inference.
    """

    video_path = UPLOAD_DIR / stored_filename

    if not video_path.exists():
        raise HTTPException(
            status_code=404,
            detail="Uploaded video was not found.",
        )

    try:
        frame_paths = extract_frames(
            video_path=video_path,
            sample_every_n_frames=30,
            max_frames=50,
        )

        face_detector = FaceDetector()
        deepfake_detector = DeepfakeDetector()

        frame_results: list[dict] = []
        all_face_paths: list[str] = []
        total_faces = 0

        for frame_path in frame_paths:
            faces = face_detector.detect_faces(frame_path)
            total_faces += len(faces)

            cropped_paths = crop_faces(
                image_path=frame_path,
                faces=faces,
            )

            all_face_paths.extend(str(path) for path in cropped_paths)

            frame_results.append(
                {
                    "frame": str(frame_path),
                    "face_count": len(faces),
                    "face_crops": [str(path) for path in cropped_paths],
                    "faces": [
                        {
                            "x": x,
                            "y": y,
                            "width": width,
                            "height": height,
                        }
                        for x, y, width, height in faces
                    ],
                }
            )

        prediction = deepfake_detector.predict(all_face_paths)

    except (FileNotFoundError, ValueError, RuntimeError) as error:
        raise HTTPException(
            status_code=400,
            detail=str(error),
        ) from error

    return {
        "status": "analysis_complete",
        "stored_filename": stored_filename,
        "frames_analyzed": len(frame_results),
        "total_faces": total_faces,
        "face_crops_created": len(all_face_paths),
        "prediction": prediction,
        "results": frame_results,
    }