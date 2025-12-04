"""
合并视频和字幕

ffmpeg -i input.mp4 -i watermark.png -filter_complex "overlay=x=10:y=10" output.mp4
"""

from pprint import pprint
import subprocess
from pathlib import Path
from typing import Optional, Union
import tempfile
import os


def merge_video_with_subtitle(
    video_path: Path,
    output_path: Optional[Path] = None,
    watermark: Optional[Path] = None,
    watermark_position: str = "bottom_right",
    watermark_opacity: float = 0.5,
) -> Path:
    if output_path is None:
        output_path = (
            video_path.parent / f"{video_path.stem}_watermark_{video_path.suffix}"
        )
    if isinstance(watermark, str):
        watermark = Path(watermark)
    watermark_path = watermark
    command = [
        "ffmpeg",
        "-hwaccel",
        "cuda",
        "-i",
        str(video_path),
    ]
    if watermark_path and watermark_path.exists():
        position_map = {
            "top_left": "x=10:y=10",
            "top_right": "x=W-w-10:y=10",
            "bottom_left": "x=10:y=H-h-10",
            "bottom_right": "x=W-w-10:y=H-h-10",
            "center": "x=(W-w)/2:y=(H-h)/2",
        }
        overlay_position = position_map.get(
            watermark_position, position_map["bottom_right"]
        )

        command.extend(
            [
                "-i",
                str(watermark_path),
                "-filter_complex",
                f"overlay={overlay_position}",
            ]
        )
    command.append(str(output_path))

    try:
        subprocess.run(command, check=True)
    finally:
        ...
    return output_path


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Merge video with subtitle")
    parser.add_argument("video_path", type=Path, help="Path to the video file")
    parser.add_argument(
        "--output_path", type=Path, default=None, help="Path to the output file"
    )
    parser.add_argument(
        "--watermark",
        type=str,
        default=None,
        help="Watermark: either a path to image file (PNG, JPG, etc.) or text string to generate text watermark",
    )
    parser.add_argument(
        "--watermark_position",
        type=str,
        default="bottom_right",
        choices=["top_left", "top_right", "bottom_left", "bottom_right", "center"],
        help="Watermark position (default: bottom_right)",
    )
    parser.add_argument(
        "--watermark_opacity",
        type=float,
        default=0.5,
        help="Watermark opacity, range 0.0-1.0 (default: 0.5)",
    )

    args = parser.parse_args()

    merged_path = merge_video_with_subtitle(
        video_path=args.video_path,
        output_path=args.output_path,
        watermark=args.watermark,
        watermark_position=args.watermark_position,
        watermark_opacity=args.watermark_opacity,
    )
    pprint(f"Merged video saved to: {merged_path}")
