"""
# 视频处理流程

1. 获取视频文件，可以是上传的文件或 URL 指定的视频文件。
2. 检测视频文件是否包含音频轨道。
3. 如果没有音频轨道，为视频添加白噪音轨道并上传OSS返回URL。
4. 如果有音频轨道，提取音频。
5. 对音频进行MSST处理，若MSST API不可用，则跳过。
6. 对vocals音频片段进行SRTGenerator流程，生成字幕文件。
7. 将字幕文件和视频合并，并上传OSS返回URL。
"""

import os
import re
import asyncio
from contextlib import asynccontextmanager
import uuid
import zipfile
from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple
from datetime import datetime
from enum import Enum
import shutil
import aiohttp
import moviepy as mpy
import numpy as np

from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks, Query
from fastapi.responses import FileResponse, RedirectResponse
from pydantic import BaseModel, Field
import requests
import uvicorn

from src.srt import SRTGenerator

ROOT_DIR = Path(__file__).parent.resolve()
UPLOAD_DIR = Path(ROOT_DIR / "uploads")
OUTPUT_DIR = Path(ROOT_DIR / "outputs")
CLEANUP_AFTER_DOWNLOAD = False
CLEANUP_UPLOAD_FILES = False
MSST_API_URL = "http://localhost:25569"


class TaskStatus(str, Enum):
    """任务状态"""

    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class TaskInfo(BaseModel):
    """任务信息"""

    taskId: str
    status: TaskStatus
    filename: str
    created_at: datetime
    completed_at: Optional[datetime] = None
    error_message: Optional[str] = None
    result: Optional[Dict[str, Any]] = None
    sample_rate: int = Field(default=16000)
    video_data: Optional[bytes] = None
    font_size: Optional[int] = 10


class ProcessingResponse(BaseModel):
    """处理响应"""

    taskId: str
    status: TaskStatus
    message: str


class TaskResult(BaseModel):
    """任务结果"""

    taskId: str
    status: TaskStatus
    filename: str
    total_sentences: Optional[int] = None
    total_vad_segments: Optional[int] = None
    audio_duration_ms: Optional[int] = None
    srt_download_url: Optional[str] = None
    segments_download_url: Optional[str] = None
    error_message: Optional[str] = None


tasks: Dict[str, TaskInfo] = {}
generator: SRTGenerator = SRTGenerator(model_config=ModelConfig(device="cuda:1"))


@asynccontextmanager
async def preload(app: FastAPI):
    """预加载模型和处理文件夹"""
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    generator.load_models()
    yield


app = FastAPI(lifespan=preload)


@app.get("/")
async def _():
    return RedirectResponse(url="/docs")


@app.post("/api/v1/process")
async def _(
    background_tasks: BackgroundTasks,
    file: Optional[UploadFile] = File(default=None, description="视频文件"),
    video_url: Optional[str] = Query(default=None, description="视频文件 URL"),
    font_size: Optional[int] = Query(default=10, description="字幕字体大小")    
):
    """处理音频文件"""
    video_data = None
    filename = None
    task_id = str(uuid.uuid4())
    task_upload_dir = UPLOAD_DIR / task_id
    task_upload_dir.mkdir(parents=True, exist_ok=True)

    #! 1. 获取视频文件，可以是上传的文件或 URL 指定的视频文件。

    # 只上传了文件
    if file and video_url is None:
        video_data = await file.read()
        filename = file.filename
    # 只提供了视频的直链
    elif video_url and file is None:
        import aiohttp

        filename = video_url.split("/")[-1]
        upload_path = Path(task_upload_dir / filename)

        async with aiohttp.ClientSession() as session:
            async with session.get(video_url) as resp:
                if resp.status != 200:
                    raise HTTPException(status_code=400, detail="无法下载视频文件")
                with upload_path.open("wb") as f:
                    while True:
                        chunk = await resp.content.read(1024)
                        if not chunk:
                            break
                        f.write(chunk)

        video_data = upload_path.read_bytes()
    elif video_url is None and file is None:
        raise HTTPException(status_code=400, detail="必须提供视频文件或视频 URL")
    elif video_url and file:
        raise HTTPException(status_code=400, detail="只能提供视频文件或视频 URL 其一")

    filename = filename or task_id
    video_data = video_data or b""

    if video_data == b"":
        raise HTTPException(status_code=400, detail="视频文件不能为空")

    task_info = TaskInfo(
        taskId=task_id,
        status=TaskStatus.PENDING,
        filename=filename,
        created_at=datetime.now(),
        video_data=video_data,
        font_size=font_size,
    )
    tasks[task_id] = task_info

    background_tasks.add_task(
        process,
        taskId=task_id,
        video_data=video_data,
    )

    return {
        "taskId": task_id,
    }


@app.get("/api/v1/query", response_model=TaskResult)
async def get_task_status(task_id: str):
    """获取任务状态"""
    if task_id not in tasks:
        raise HTTPException(status_code=404, detail="任务不存在")

    task = tasks[task_id]
    result = TaskResult(
        taskId=task.taskId,
        status=task.status,
        filename=task.filename,
        error_message=task.error_message,
    )
    return result


@app.get("/api/v1/download/")
async def download_result(task_id: str):
    """下载处理结果"""
    try:
        if task_id not in tasks:
            raise HTTPException(status_code=404, detail="任务不存在")

        task = tasks[task_id]
        if task.status != TaskStatus.COMPLETED or not task.result:
            return {"status": task.status, "message": "任务尚未完成"}

        result_url = task.result.get("url")
        if not result_url:
            raise HTTPException(status_code=400, detail="无有效的下载链接")

        return {"url": result_url, "status": task.status}
    except Exception as e:
        return {"status": "failed", "error_message": str(e)}


async def generate_white_noise(duration, fps=16000):
    """生成指定时长的白噪音音频"""
    samples = int(duration * fps)
    noise = np.random.normal(0, 0.1, (samples, 2)).astype(np.float32)
    return noise


async def download_file(url, output_path):
    """下载文件"""
    response = requests.get(url, stream=True)
    if response.status_code == 404:
        parts = url.rsplit("/", 1)
        if len(parts) == 2:
            base_id = parts[1]
            response = requests.get(url, stream=True)

    response.raise_for_status()
    with open(output_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
    print(f"已下载: {output_path}")


async def generate_srt_file(segments: List[Tuple[str, List[int]]], output_path: Path):
    """生成 SRT 文件"""

    def ms_to_srt_time(ms: int) -> str:
        total_ms = int(round(ms))
        hours = total_ms // 3600000
        remaining = total_ms % 3600000
        minutes = remaining // 60000
        remaining = remaining % 60000
        seconds = remaining // 1000
        milliseconds = remaining % 1000
        return f"{hours:02d}:{minutes:02d}:{seconds:02d},{milliseconds:03d}"

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        for idx, (text, timestamp) in enumerate(segments, 1):
            # 如果 text 过长则进行换行
            threhold = 15
            if len(text) > threhold:
                # 每40个字符换行一次
                text = "\n".join(
                    [text[i : i + threhold] for i in range(0, len(text), threhold)]
                )
            start_time_str = ms_to_srt_time(timestamp[0])
            end_time_str = ms_to_srt_time(timestamp[1])
            f.write(f"{idx}\n")
            f.write(f"{start_time_str} --> {end_time_str}\n")
            f.write(f"{text}\n\n")

    print(f"\nSRT 文件已生成: {output_path}")


async def process(
    taskId: str,
    video_data: bytes,
):
    """视频 文件处理任务"""
    if not generator.model_manager._loaded:
        return HTTPException(status_code=500, detail="模型未加载")

    task = tasks[taskId]
    task.status = TaskStatus.PROCESSING
    print(f"开始处理任务: {taskId}")
    output_dir = OUTPUT_DIR / taskId
    output_dir.mkdir(parents=True, exist_ok=True)
    video_path = OUTPUT_DIR / f"{taskId}_input_video.mp4"
    with video_path.open("wb") as f:
        f.write(video_data)
    clip = mpy.VideoFileClip(str(video_path))

    has_valid_audio = False
    if clip.audio is not None:
        check_duration = min(5.0, clip.duration)
        audio_array = clip.audio.to_soundarray(fps=16000, nbytes=2)[
            : int(check_duration * 16000)
        ]
        rms = np.sqrt(np.mean(audio_array**2))
        has_valid_audio = rms > 0.0
        print(
            f"音频检测: task_id={taskId}, RMS={rms:.6f}, 有效音频={'是' if has_valid_audio else '否'}"
        )

    if not has_valid_audio:
        #! 添加白噪音音轨
        reason = "视频无音频轨道" if clip.audio is None else "视频音频为静音"
        print(f"{reason}，为视频添加白噪音: {taskId}")
        duration = clip.duration
        white_noise = await generate_white_noise(duration, fps=16000)
        audio_clip = mpy.AudioArrayClip(white_noise, fps=16000)
        clip = clip.with_audio(audio_clip)
        video_with_noise_path = OUTPUT_DIR / f"{taskId}_with_noise.mp4"
        clip.write_videofile(
            str(video_with_noise_path), codec="libx264", audio_codec="aac", logger=None
        )
        print(f"白噪音添加完成: {taskId}")
        print(f"含白噪音视频已保存: {video_with_noise_path}")
        clip.close()

        res = await put_file(video_with_noise_path)
        task.status = TaskStatus.COMPLETED
        task.result = {"url": res[1]}
        return task.result
    else:
        #! SRT
        print(f"视频含音频轨道，提取音频: {taskId}")
        audio_path = OUTPUT_DIR / f"{taskId}_audio.wav"
        clip = mpy.VideoFileClip(video_path)
        clip.audio.write_audiofile(filename=audio_path, fps=16000)

        #! MSST
        print(f"开始 MSST 处理: {taskId}")
        async with aiohttp.ClientSession() as session:
            f = open(audio_path, "rb")
            try:
                form = aiohttp.FormData()
                form.add_field(
                    "file",
                    f,
                    filename=audio_path.name,
                    content_type="audio/wav",
                )
                async with session.post(
                    MSST_API_URL + "/v1/audio/separations", data=form
                ) as resp:
                    if resp.status == 200:
                        msst_result = await resp.json()
                        print(f"MSST 处理完成: {taskId}, 结果: {msst_result}")
                    else:
                        task.status = TaskStatus.FAILED
                        task.error_message = (
                            f"MSST 处理失败: {taskId}, 状态码: {resp.status}"
                        )
                        return {"error_message": task.error_message}
            finally:
                f.close()

        vocals_url = msst_result.get("data").get("files").get("vocals").get("url")
        audio_path = OUTPUT_DIR / f"{taskId}_vocals.wav"
        print(f"开始下载 vocals 音频: {taskId}, 地址: {vocals_url}, 路径: {audio_path}")
        await download_file(MSST_API_URL + vocals_url, audio_path)
        print(f"已下载 vocals 音频: {audio_path}")
        srt_result = generator(audio_path=audio_path, output_dir=OUTPUT_DIR)
        print(f"SRT 生成完成: {taskId}")
        srt_file_path = OUTPUT_DIR / f"srt/{taskId}.srt"
        await generate_srt_file(srt_result.segments, srt_file_path)
        #! 合并SRT与视频
        if srt_file_path.exists():
            print(f"开始合并 SRT 与视频: {taskId}")
            # ffmpeg -y -i input.mp4 -vf "subtitles=subtitle.srt:force_style='Alignment=2,MarginV=5'" -c:a copy output.mp4
            # 字幕字体缩小：添加 Fontsize 样式参数，可按需再调整数值
            command = [
                "ffmpeg",
                "-y",
                "-hwaccel",
                "cuda",
                "-i",
                video_path,
                "-vf",
                f"subtitles={srt_file_path}:force_style='Alignment=2,MarginV=5,Fontsize={task.font_size}'",
                "-c:a",
                "copy",
                output_dir / f"{taskId}_final_video.mp4",
            ]
            print(command)
            process = await asyncio.create_subprocess_exec(
                *command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await process.communicate()
            if process.returncode == 0:
                print(f"SRT 与视频合并完成: {taskId}")
                final_video_path = output_dir / f"{taskId}_final_video.mp4"
                res = await put_file(final_video_path)
                task.status = TaskStatus.COMPLETED
                task.result = {
                    "url": res[1],
                }
                print(f"任务完成: {taskId}, 结果: {task.result}")
                return task.result
        else:
            task.status = TaskStatus.FAILED
            return {"error_message": "SRT 文件生成失败"}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=25561)
