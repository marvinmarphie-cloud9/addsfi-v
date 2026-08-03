from pathlib import Path

from fastapi import APIRouter, HTTPException
from torch import nn
from torchvision.models import efficientnet_b0

from addfs.inference.video_predictor import VideoPredictor


router = APIRouter(
    prefix="/prediction",
    tags=["Deepfake Prediction"],
)

PROJECT_ROOT = Path(__file__).resolve().parents[3]
UPLOAD_DIR = PROJECT_ROOT / "uploads"


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
        predictor = VideoPredictor(
            model=create_model(),
            checkpoint_path=find_latest_checkpoint(),
            frame_interval=30,
        )

        return predictor.predict(video_path)

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