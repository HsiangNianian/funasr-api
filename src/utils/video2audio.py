"""
视频音频提取
"""

import argparse
import os
import sys
import yt_dlp
from pathlib import Path
from moviepy import VideoFileClip


def download_video(url: str, output_dir: str = "downloads") -> str | None:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    ydl_opts = {
        "format": "bestvideo[ext=mp4]+bestaudio[ext=m4a]/bestvideo+bestaudio/best",
        "outtmpl": str(output_path / "%(title)s.%(ext)s"),
        "quiet": False,
        "no_warnings": False,
        "ignoreerrors": False,
        "merge_output_format": "mp4",
    }
    if "douyin.com" in url or "tiktok.com" in url:
        # 抖音需要从浏览器获取cookies
        ydl_opts["cookiesfrombrowser"] = ("chrome",)
    elif "bilibili.com" in url or "b23.tv" in url:
        ydl_opts["format"] = "bestvideo[height<=1080]+bestaudio/best[height<=1080]"

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)

            if info:
                video_file = ydl.prepare_filename(info)
                return video_file
            else:
                raise ValueError("无法提取视频信息")

    except Exception as e:
        print(f"下载视频时出错: {e}")
        return None


def extract_audio_from_video(
    video_path: str, output_dir: str = "outputs/audio", audio_format: str = "wav"
) -> str | None:
    """
    从视频文件提取音频
    """
    if not os.path.exists(video_path):
        print(f"视频文件不存在: {video_path}")
        return None

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    video_name = Path(video_path).stem
    audio_file = output_path / f"{video_name}.{audio_format}"

    try:
        video_clip = VideoFileClip(video_path)

        if video_clip.audio is None:
            # 没有音轨的情况
            video_clip.close()
            return None

        if audio_format.lower() == "wav":
            video_clip.audio.write_audiofile(
                str(audio_file), codec="pcm_s16le", logger=None
            )
        else:
            video_clip.audio.write_audiofile(str(audio_file), logger=None)

        video_clip.close()

        return str(audio_file)

    except Exception as e:
        print(f"提取音频时出错: {e}")
        return None


def process_video_url(
    url: str,
    download_dir: str = "downloads",
    audio_dir: str = "outputs/audio",
    audio_format: str = "wav",
    keep_video: bool = False,
) -> str | None:
    # 下载视频
    video_path = download_video(url, download_dir)
    if not video_path:
        return None

    # 提取音频
    audio_path = extract_audio_from_video(video_path, audio_dir, audio_format)
    if not audio_path:
        if not keep_video and os.path.exists(video_path):
            os.remove(video_path)
        return None

    return audio_path


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-u",
        "--url",
        required=True,
    )

    parser.add_argument(
        "-d",
        "--download-dir",
        default="downloads",
    )

    parser.add_argument(
        "-o",
        "--output",
        default="outputs/audio",
    )

    parser.add_argument(
        "-f",
        "--format",
        default="wav",
        choices=["wav", "mp3", "ogg", "m4a", "flac"],
    )

    parser.add_argument(
        "--keep-video",
        action="store_true",
    )

    args = parser.parse_args()

    try:
        result = process_video_url(
            url=args.url,
            download_dir=args.download_dir,
            audio_dir=args.output,
            audio_format=args.format,
            keep_video=args.keep_video,
        )

        if result:
            return 0
        else:
            return 1

    except Exception as e:
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
