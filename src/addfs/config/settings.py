from pathlib import Path
from dotenv import load_dotenv
import os

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parents[3]

FACEFORENSICS_ROOT = Path(
    os.getenv(
        "FACEFORENSICS_ROOT",
        PROJECT_ROOT / "datasets"
    )
)

PREPARED_DATASET = Path(
    os.getenv(
        "PREPARED_DATASET",
        PROJECT_ROOT / "datasets" / "prepared"
    )
)

RESULTS_DIR = Path(
    os.getenv(
        "RESULTS_DIR",
        PROJECT_ROOT / "results"
    )
)

MODELS_DIR = Path(
    os.getenv(
        "MODELS_DIR",
        PROJECT_ROOT / "models"
    )
)

UPLOADS_DIR = Path(
    os.getenv(
        "UPLOADS_DIR",
        PROJECT_ROOT / "uploads"
    )
)

TEMP_DIR = Path(
    os.getenv(
        "TEMP_DIR",
        PROJECT_ROOT / "temp"
    )
)

BATCH_SIZE = int(os.getenv("BATCH_SIZE", 16))
NUM_WORKERS = int(os.getenv("NUM_WORKERS", 4))
LEARNING_RATE = float(os.getenv("LEARNING_RATE", 0.001))
EPOCHS = int(os.getenv("EPOCHS", 25))

CONFIDENCE_THRESHOLD = float(
    os.getenv("CONFIDENCE_THRESHOLD", 0.5)
)