from pathlib import Path
from uuid import uuid4

import cv2


def extract_frames(
    video_path: str | Path,
    output_root: str | Path = "temp/frames",
    sample_every_n_frames: int = 30,
    max_frames: int = 100,
) -> list[Path]:
    """
    Extract sampled frames from a video and return their saved paths.

    Args:
        video_path: Path to the input video.
        output_root: Parent directory for extracted frames.
        sample_every_n_frames: Save one frame after every N frames.
        max_frames: Maximum number of frames to save.
    """

    if sample_every_n_frames <= 0:
        raise ValueError("sample_every_n_frames must be greater than zero.")

    if max_frames <= 0:
        raise ValueError("max_frames must be greater than zero.")

    video_path = Path(video_path)

    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    output_dir = Path(output_root) / uuid4().hex
    output_dir.mkdir(parents=True, exist_ok=True)

    capture = cv2.VideoCapture(str(video_path))

    if not capture.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    saved_frames: list[Path] = []
    frame_index = 0

    try:
        while len(saved_frames) < max_frames:
            success, frame = capture.read()

            if not success:
                break

            if frame_index % sample_every_n_frames == 0:
                frame_path = output_dir / f"frame_{frame_index:06d}.jpg"
                written = cv2.imwrite(str(frame_path), frame)

                if written:
                    saved_frames.append(frame_path)

            frame_index += 1
    finally:
        capture.release()

    if not saved_frames:
        raise ValueError("No readable frames were extracted from the video.")

    return saved_frames