from pathlib import Path
from tempfile import NamedTemporaryFile

from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from torch import nn
from torchvision.models import efficientnet_b0

from addfs.inference.video_predictor import VideoPredictor


router = APIRouter(
    prefix="/prediction",
    tags=["Deepfake Prediction"],
)

PROJECT_ROOT = Path(__file__).resolve().parents[3]
FRONTEND_FILE = PROJECT_ROOT / "frontend" / "index.html"
UPLOAD_DIR = PROJECT_ROOT / "uploads"

ALLOWED_EXTENSIONS = {
    ".mp4",
    ".mov",
    ".avi",
    ".mkv",
}

MAX_FILE_SIZE = 500 * 1024 * 1024


def find_latest_checkpoint() -> Path:
    checkpoints = sorted(
        PROJECT_ROOT.glob(
            "results/training/run_*/best_model.pt"
        ),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )

    if not checkpoints:
        raise FileNotFoundError(
            "No trained model checkpoint was found."
        )

    return checkpoints[0]


def create_model() -> nn.Module:
    model = efficientnet_b0(weights=None)

    input_features = model.classifier[1].in_features

    model.classifier[1] = nn.Linear(
        input_features,
        2,
    )

    return model


def create_predictor() -> VideoPredictor:
    return VideoPredictor(
        model=create_model(),
        checkpoint_path=find_latest_checkpoint(),
        frame_interval=30,
    )


@router.post("/video/{stored_filename}")
def predict_uploaded_video(
    stored_filename: str,
) -> dict:
    video_path = UPLOAD_DIR / stored_filename

    if not video_path.exists():
        raise HTTPException(
            status_code=404,
            detail="Uploaded video was not found.",
        )

    try:
        return create_predictor().predict(video_path)

    except FileNotFoundError as error:
        raise HTTPException(
            status_code=503,
            detail=str(error),
        ) from error

    except (RuntimeError, ValueError) as error:
        raise HTTPException(
            status_code=400,
            detail=str(error),
        ) from error

@router.get("/ui", include_in_schema=False)
def prediction_interface() -> FileResponse:
    if not FRONTEND_FILE.exists():
        raise HTTPException(
            status_code=404,
            detail="Frontend file was not found.",
        )

    return FileResponse(FRONTEND_FILE)

@router.post("/video-upload")
async def upload_and_predict_video(
    file: UploadFile = File(...),
) -> dict:
    original_filename = file.filename or "uploaded-video"

    extension = Path(
        original_filename
    ).suffix.lower()

    if extension not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=(
                "Unsupported file type. "
                "Use MP4, MOV, AVI, or MKV."
            ),
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
            detail="The video exceeds the 500 MB limit.",
        )

    temporary_path: Path | None = None

    try:
        with NamedTemporaryFile(
            delete=False,
            suffix=extension,
        ) as temporary_file:
            temporary_file.write(file_bytes)

            temporary_path = Path(
                temporary_file.name
            )

        result = create_predictor().predict(
            temporary_path
        )

        result["original_filename"] = (
            original_filename
        )

        result["size_bytes"] = len(file_bytes)

        return result

    except FileNotFoundError as error:
        raise HTTPException(
            status_code=503,
            detail=str(error),
        ) from error

    except (RuntimeError, ValueError) as error:
        raise HTTPException(
            status_code=400,
            detail=str(error),
        ) from error

    finally:
        if (
            temporary_path is not None
            and temporary_path.exists()
        ):
            temporary_path.unlink()