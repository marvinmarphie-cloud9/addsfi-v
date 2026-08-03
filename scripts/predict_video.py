from pathlib import Path
import argparse
import sys

from torch import nn
from torchvision.models import efficientnet_b0

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from addfs.inference.video_predictor import VideoPredictor


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
            "No trained checkpoint was found."
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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Classify a video as real or fake."
    )

    parser.add_argument(
        "video_path",
        type=Path,
        help="Path to the video.",
    )

    parser.add_argument(
        "--frame-interval",
        type=int,
        default=30,
        help="Analyze one frame every N frames.",
    )

    args = parser.parse_args()

    predictor = VideoPredictor(
        model=create_model(),
        checkpoint_path=find_latest_checkpoint(),
        frame_interval=args.frame_interval,
    )

    result = predictor.predict(args.video_path)

    print(f"Video: {result['video']}")
    print(f"Prediction: {result['prediction']}")
    print(f"Confidence: {result['confidence']:.2%}")
    print(
        f"Real probability: "
        f"{result['real_probability']:.2%}"
    )
    print(
        f"Fake probability: "
        f"{result['fake_probability']:.2%}"
    )
    print(
        f"Frames analyzed: "
        f"{result['frames_analyzed']}"
    )
    print(
        f"Faces analyzed: "
        f"{result['faces_analyzed']}"
    )


if __name__ == "__main__":
    main()