import os
from pathlib import Path

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[3]

load_dotenv(PROJECT_ROOT / ".env")


def get_path(
    variable_name: str,
    default: Path,
) -> Path:
    raw_value = os.getenv(variable_name)

    if not raw_value:
        return default.resolve()

    configured_path = Path(raw_value).expanduser()

    if configured_path.is_absolute():
        return configured_path.resolve()

    return (PROJECT_ROOT / configured_path).resolve()


def get_int(
    variable_name: str,
    default: int,
    minimum: int | None = None,
) -> int:
    raw_value = os.getenv(variable_name)

    try:
        value = int(raw_value) if raw_value else default
    except ValueError as error:
        raise ValueError(
            f"{variable_name} must be an integer."
        ) from error

    if minimum is not None and value < minimum:
        raise ValueError(
            f"{variable_name} must be at least {minimum}."
        )

    return value


def get_float(
    variable_name: str,
    default: float,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    raw_value = os.getenv(variable_name)

    try:
        value = float(raw_value) if raw_value else default
    except ValueError as error:
        raise ValueError(
            f"{variable_name} must be a number."
        ) from error

    if minimum is not None and value < minimum:
        raise ValueError(
            f"{variable_name} must be at least {minimum}."
        )

    if maximum is not None and value > maximum:
        raise ValueError(
            f"{variable_name} must not exceed {maximum}."
        )

    return value


FACEFORENSICS_ROOT = get_path(
    "FACEFORENSICS_ROOT",
    PROJECT_ROOT / "datasets" / "faceforensics",
)

PREPARED_DATASET = get_path(
    "PREPARED_DATASET",
    PROJECT_ROOT / "datasets" / "prepared",
)

WORKING_DATASET = get_path(
    "WORKING_DATASET",
    PROJECT_ROOT / "datasets" / "working",
)

RESULTS_DIR = get_path(
    "RESULTS_DIR",
    PROJECT_ROOT / "results",
)

MODELS_DIR = get_path(
    "MODELS_DIR",
    PROJECT_ROOT / "models",
)

UPLOADS_DIR = get_path(
    "UPLOADS_DIR",
    PROJECT_ROOT / "uploads",
)

TEMP_DIR = get_path(
    "TEMP_DIR",
    PROJECT_ROOT / "temp",
)

BATCH_SIZE = get_int(
    "BATCH_SIZE",
    default=16,
    minimum=1,
)

NUM_WORKERS = get_int(
    "NUM_WORKERS",
    default=0,
    minimum=0,
)

EPOCHS = get_int(
    "EPOCHS",
    default=25,
    minimum=1,
)

FRAME_INTERVAL = get_int(
    "FRAME_INTERVAL",
    default=30,
    minimum=1,
)

IMAGE_SIZE = get_int(
    "IMAGE_SIZE",
    default=224,
    minimum=32,
)

EARLY_STOPPING_PATIENCE = get_int(
    "EARLY_STOPPING_PATIENCE",
    default=5,
    minimum=1,
)

SCHEDULER_PATIENCE = get_int(
    "SCHEDULER_PATIENCE",
    default=2,
    minimum=0,
)

LEARNING_RATE = get_float(
    "LEARNING_RATE",
    default=0.001,
    minimum=0.0000001,
)

CONFIDENCE_THRESHOLD = get_float(
    "CONFIDENCE_THRESHOLD",
    default=0.5,
    minimum=0.0,
    maximum=1.0,
)

MAX_UPLOAD_SIZE_MB = get_int(
    "MAX_UPLOAD_SIZE_MB",
    default=500,
    minimum=1,
)

MAX_UPLOAD_SIZE_BYTES = (
    MAX_UPLOAD_SIZE_MB * 1024 * 1024
)


def create_runtime_directories() -> None:
    directories = [
        PREPARED_DATASET,
        WORKING_DATASET,
        RESULTS_DIR,
        MODELS_DIR,
        UPLOADS_DIR,
        TEMP_DIR,
    ]

    for directory in directories:
        directory.mkdir(
            parents=True,
            exist_ok=True,
        )


def validate_configuration() -> list[str]:
    warnings: list[str] = []

    if not FACEFORENSICS_ROOT.exists():
        warnings.append(
            "FaceForensics++ directory was not found: "
            f"{FACEFORENSICS_ROOT}"
        )

    return warnings