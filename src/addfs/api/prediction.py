from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

from fastapi import (
    APIRouter,
    File,
    HTTPException,
    UploadFile,
)
from fastapi.responses import (
    FileResponse,
    Response,
)
from torch import nn
from torchvision.models import efficientnet_b0

from addfs.config import (
    CONFIDENCE_THRESHOLD,
    FRAME_INTERVAL,
    MAX_UPLOAD_SIZE_BYTES,
    MAX_UPLOAD_SIZE_MB,
    PROJECT_ROOT,
    RESULTS_DIR,
    TEMP_DIR,
    UPLOADS_DIR,
    create_runtime_directories,
)
from addfs.inference.video_predictor import (
    VideoPredictor,
)
from addfs.reporting.report_generator import (
    generate_prediction_report,
)


router = APIRouter(
    prefix="/prediction",
    tags=["Deepfake Prediction"],
)

FRONTEND_FILE = (
    PROJECT_ROOT
    / "frontend"
    / "index.html"
)

ALLOWED_EXTENSIONS = {
    ".mp4",
    ".mov",
    ".avi",
    ".mkv",
}


def find_latest_checkpoint() -> Path:
    checkpoints = sorted(
        RESULTS_DIR.glob(
            "training/run_*/best_model.pt"
        ),
        key=lambda path: (
            path.stat().st_mtime
        ),
        reverse=True,
    )

    if not checkpoints:
        raise FileNotFoundError(
            "No trained model checkpoint "
            "was found under "
            f"{RESULTS_DIR / 'training'}."
        )

    return checkpoints[0]


def create_model() -> nn.Module:
    model = efficientnet_b0(
        weights=None
    )

    input_features = (
        model.classifier[1].in_features
    )

    model.classifier[1] = nn.Linear(
        input_features,
        2,
    )

    return model


def create_predictor() -> VideoPredictor:
    checkpoint_path = (
        find_latest_checkpoint()
    )

    return VideoPredictor(
        model=create_model(),
        checkpoint_path=checkpoint_path,
        frame_interval=FRAME_INTERVAL,
        threshold=CONFIDENCE_THRESHOLD,
    )


@router.get(
    "/ui",
    include_in_schema=False,
)
def prediction_interface() -> FileResponse:
    if not FRONTEND_FILE.exists():
        raise HTTPException(
            status_code=404,
            detail=(
                "Frontend file was not found."
            ),
        )

    return FileResponse(
        FRONTEND_FILE
    )


@router.post(
    "/video/{stored_filename}"
)
def predict_uploaded_video(
    stored_filename: str,
) -> dict:
    create_runtime_directories()

    video_path = (
        UPLOADS_DIR
        / stored_filename
    )

    if not video_path.exists():
        raise HTTPException(
            status_code=404,
            detail=(
                "Uploaded video was not found."
            ),
        )

    try:
        return (
            create_predictor()
            .predict(video_path)
        )

    except FileNotFoundError as error:
        raise HTTPException(
            status_code=503,
            detail=str(error),
        ) from error

    except (
        RuntimeError,
        ValueError,
    ) as error:
        raise HTTPException(
            status_code=400,
            detail=str(error),
        ) from error


@router.post("/video-upload")
async def upload_and_predict_video(
    file: UploadFile = File(...),
) -> dict:
    create_runtime_directories()

    original_filename = (
        file.filename
        or "uploaded-video"
    )

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
            detail=(
                "The uploaded file is empty."
            ),
        )

    if (
        len(file_bytes)
        > MAX_UPLOAD_SIZE_BYTES
    ):
        raise HTTPException(
            status_code=413,
            detail=(
                "The video exceeds the "
                f"configured "
                f"{MAX_UPLOAD_SIZE_MB} MB "
                "limit."
            ),
        )

    temporary_path: (
        Path | None
    ) = None

    try:
        with NamedTemporaryFile(
            delete=False,
            suffix=extension,
            dir=TEMP_DIR,
        ) as temporary_file:
            temporary_file.write(
                file_bytes
            )

            temporary_path = Path(
                temporary_file.name
            )

        checkpoint_path = (
            find_latest_checkpoint()
        )

        predictor = VideoPredictor(
            model=create_model(),
            checkpoint_path=(
                checkpoint_path
            ),
            frame_interval=(
                FRAME_INTERVAL
            ),
            threshold=(
                CONFIDENCE_THRESHOLD
            ),
        )

        result = predictor.predict(
            temporary_path
        )

        result[
            "original_filename"
        ] = original_filename

        result[
            "size_bytes"
        ] = len(file_bytes)

        result[
            "model_checkpoint"
        ] = str(checkpoint_path)

        return result

    except FileNotFoundError as error:
        raise HTTPException(
            status_code=503,
            detail=str(error),
        ) from error

    except (
        RuntimeError,
        ValueError,
    ) as error:
        raise HTTPException(
            status_code=400,
            detail=str(error),
        ) from error

    finally:
        if (
            temporary_path
            is not None
            and temporary_path.exists()
        ):
            temporary_path.unlink()


@router.post("/report")
def download_prediction_report(
    payload: dict[str, Any],
) -> Response:
    try:
        pdf_bytes = (
            generate_prediction_report(
                payload
            )
        )

        return Response(
            content=pdf_bytes,
            media_type=(
                "application/pdf"
            ),
            headers={
                "Content-Disposition": (
                    "attachment; "
                    'filename="'
                    "ADDFS_Analysis_Report.pdf"
                    '"'
                )
            },
        )

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=(
                "Unable to generate "
                "the PDF report: "
                f"{error}"
            ),
        ) from error