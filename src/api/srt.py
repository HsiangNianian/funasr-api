"""
SRT 字幕生成 API

提供音频文件转 SRT 字幕的 RESTful API，支持异步任务处理
"""

from __future__ import annotations

import asyncio
import shutil
import uuid
import zipfile
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import (
    APIRouter,
    BackgroundTasks,
    File,
    Form,
    HTTPException,
    Query,
    Request,
    UploadFile,
)
from fastapi.responses import FileResponse, PlainTextResponse
from pydantic import BaseModel, Field

from src.log import logger
from src.srt import ProcessingResult, SRTConfig, SRTGenerator

router = APIRouter(prefix="/srt", tags=["SRT 字幕生成"])

# =============================================================================
# 配置
# =============================================================================

# 文件存储目录（相对于项目根目录）
UPLOAD_DIR = Path("uploads/srt")
OUTPUT_DIR = Path("outputs/srt")

# 清理策略
CLEANUP_AFTER_DOWNLOAD = False  # 下载后是否清理文件
CLEANUP_UPLOAD_FILES = True  # 处理完成后是否清理上传文件


# =============================================================================
# 枚举与数据模型
# =============================================================================


class TaskStatus(str, Enum):
    """任务状态"""

    PENDING = "pending"  # 等待处理
    PROCESSING = "processing"  # 处理中
    COMPLETED = "completed"  # 已完成
    FAILED = "failed"  # 失败


class TaskInfo(BaseModel):
    """任务信息（内部存储用）"""

    task_id: str
    status: TaskStatus
    filename: str
    created_at: datetime
    completed_at: Optional[datetime] = None
    error_message: Optional[str] = None
    result: Optional[Dict[str, Any]] = None
    save_segments: bool = False


class CreateTaskResponse(BaseModel):
    """创建任务响应"""

    task_id: str = Field(description="任务ID")
    status: TaskStatus = Field(description="任务状态")
    message: str = Field(description="提示信息")


class TaskStatusResponse(BaseModel):
    """任务状态响应"""

    task_id: str = Field(description="任务ID")
    status: TaskStatus = Field(description="任务状态")
    filename: str = Field(description="原始文件名")
    created_at: datetime = Field(description="创建时间")
    completed_at: Optional[datetime] = Field(default=None, description="完成时间")
    error_message: Optional[str] = Field(default=None, description="错误信息")
    # 结果字段
    total_sentences: Optional[int] = Field(default=None, description="总句子数")
    total_vad_segments: Optional[int] = Field(default=None, description="VAD 分段数")
    audio_duration_ms: Optional[int] = Field(
        default=None, description="音频时长（毫秒）"
    )
    sample_rate: Optional[int] = Field(default=None, description="采样率")
    # 下载链接
    srt_download_url: Optional[str] = Field(
        default=None, description="SRT 文件下载链接"
    )
    segments_download_url: Optional[str] = Field(
        default=None, description="音频片段下载链接"
    )


class TaskListResponse(BaseModel):
    """任务列表响应"""

    total: int = Field(description="总任务数")
    filtered: int = Field(description="过滤后数量")
    tasks: List[TaskStatusResponse] = Field(description="任务列表")


class SRTContentResponse(BaseModel):
    """SRT 内容响应"""

    task_id: str
    srt_content: str
    total_sentences: int


class ServiceStatusResponse(BaseModel):
    """服务状态响应"""

    ready: bool = Field(description="服务是否就绪")
    asr_model: Optional[str] = Field(default=None, description="ASR 模型")
    vad_model: bool = Field(description="VAD 模型是否加载")
    punc_model: bool = Field(description="标点模型是否加载")
    fazh_model: bool = Field(description="时间戳模型是否加载")
    loaded_models: List[str] = Field(description="已加载的模型列表")
    pending_tasks: int = Field(description="等待中的任务数")
    processing_tasks: int = Field(description="处理中的任务数")


# =============================================================================
# 任务存储（生产环境应使用 Redis 或数据库）
# =============================================================================

_tasks: Dict[str, TaskInfo] = {}


def _get_task(task_id: str) -> TaskInfo:
    """获取任务，不存在则抛出异常"""
    if task_id not in _tasks:
        raise HTTPException(status_code=404, detail=f"任务不存在: {task_id}")
    return _tasks[task_id]


def _task_to_response(task: TaskInfo) -> TaskStatusResponse:
    """将内部任务信息转换为响应格式"""
    response = TaskStatusResponse(
        task_id=task.task_id,
        status=task.status,
        filename=task.filename,
        created_at=task.created_at,
        completed_at=task.completed_at,
        error_message=task.error_message,
    )

    if task.status == TaskStatus.COMPLETED and task.result:
        response.total_sentences = task.result.get("total_sentences")
        response.total_vad_segments = task.result.get("total_vad_segments")
        response.audio_duration_ms = task.result.get("audio_duration_ms")
        response.sample_rate = task.result.get("sample_rate")
        response.srt_download_url = f"/srt/download/{task.task_id}/srt"

        if task.result.get("has_segments"):
            response.segments_download_url = f"/srt/download/{task.task_id}/segments"

    return response


# =============================================================================
# 辅助函数
# =============================================================================


def _ensure_dirs():
    """确保目录存在"""
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def _get_srt_generator(
    request: Request, config: Optional[SRTConfig] = None
) -> SRTGenerator:
    """获取 SRT 生成器"""
    funasr_app = request.app.state.funasr_app
    return SRTGenerator(funasr_app, config)


def _validate_models(request: Request) -> None:
    """验证所需模型是否已加载"""
    funasr_app = request.app.state.funasr_app
    loaded_models = funasr_app.get_loaded_models()

    # 检查是否有 ASR 模型
    asr_models = ["paraformer_zh", "sensevoice", "paraformer_en", "whisper"]
    has_asr = any(model in loaded_models for model in asr_models)
    if not has_asr:
        raise HTTPException(
            status_code=503,
            detail="没有可用的 ASR 模型，请在配置中启用 paraformer_zh、sensevoice、paraformer_en 或 whisper",
        )

    # 检查 VAD 模型
    if "fsmnvad" not in loaded_models:
        raise HTTPException(
            status_code=503,
            detail="VAD 模型未加载，请在配置中启用 fsmnvad",
        )


async def _process_audio_task(
    request: Request,
    task_id: str,
    audio_path: Path,
    save_segments: bool,
    long_segment_threshold_ms: int,
):
    """后台处理音频任务"""
    task = _tasks.get(task_id)
    if not task:
        logger.error(f"任务不存在: {task_id}")
        return

    try:
        task.status = TaskStatus.PROCESSING
        logger.info(f"开始处理任务: {task_id}, 文件: {task.filename}")

        # 创建输出目录
        output_dir = OUTPUT_DIR / task_id
        output_dir.mkdir(parents=True, exist_ok=True)

        # 创建生成器
        config = SRTConfig(long_segment_threshold_ms=long_segment_threshold_ms)
        generator = _get_srt_generator(request, config)

        # 在线程池中执行处理
        loop = asyncio.get_event_loop()
        result: ProcessingResult = await loop.run_in_executor(
            None,
            lambda: generator.process_file(
                audio_path=audio_path,
                output_dir=output_dir,
                save_srt=True,
                save_segments=save_segments,
            ),
        )

        # 更新任务状态
        task.status = TaskStatus.COMPLETED
        task.completed_at = datetime.now()
        task.result = {
            "total_sentences": result.metadata.get("total_sentences", 0),
            "total_vad_segments": result.metadata.get("total_vad_segments", 0),
            "audio_duration_ms": result.metadata.get("audio_duration_ms", 0),
            "sample_rate": result.metadata.get("sample_rate", 16000),
            "has_segments": save_segments,
            "srt_content": result.srt_content,
        }

        logger.info(
            f"任务完成: {task_id}, 生成 {result.metadata.get('total_sentences', 0)} 条字幕"
        )

    except Exception as e:
        logger.error(f"任务处理失败: {task_id}, 错误: {e}")
        task.status = TaskStatus.FAILED
        task.completed_at = datetime.now()
        task.error_message = str(e)

    finally:
        # 清理上传文件
        if CLEANUP_UPLOAD_FILES:
            try:
                if audio_path.exists():
                    audio_path.unlink()
                # 清理上传目录
                upload_dir = UPLOAD_DIR / task_id
                if upload_dir.exists() and not any(upload_dir.iterdir()):
                    upload_dir.rmdir()
            except Exception as e:
                logger.warning(f"清理上传文件失败: {e}")


async def _cleanup_task_files(task_id: str):
    """清理任务文件"""
    try:
        # 清理输出目录
        output_dir = OUTPUT_DIR / task_id
        if output_dir.exists():
            shutil.rmtree(output_dir)

        # 清理压缩文件
        zip_path = OUTPUT_DIR / f"{task_id}_segments.zip"
        if zip_path.exists():
            zip_path.unlink()

        # 清理上传目录
        upload_dir = UPLOAD_DIR / task_id
        if upload_dir.exists():
            shutil.rmtree(upload_dir)

        logger.info(f"已清理任务文件: {task_id}")
    except Exception as e:
        logger.error(f"清理任务文件失败: {task_id}, 错误: {e}")


# =============================================================================
# API 路由 - 任务管理
# =============================================================================


@router.post(
    "/process",
    response_model=CreateTaskResponse,
    summary="提交音频处理任务",
    description="上传音频文件，创建异步处理任务",
)
async def create_process_task(
    request: Request,
    background_tasks: BackgroundTasks,
    audio: UploadFile = File(..., description="音频文件（支持 wav, mp3, flac 等格式）"),
    save_segments: bool = Form(default=False, description="是否保存音频片段"),
    long_segment_threshold_ms: int = Form(
        default=3000, description="长片段阈值（毫秒），超过此值使用 FA-ZH 细化"
    ),
) -> CreateTaskResponse:
    """提交音频处理任务"""
    _validate_models(request)
    _ensure_dirs()

    # 生成任务ID
    task_id = str(uuid.uuid4())

    # 保存上传文件
    task_upload_dir = UPLOAD_DIR / task_id
    task_upload_dir.mkdir(parents=True, exist_ok=True)

    filename = audio.filename or f"audio_{task_id}"
    upload_path = task_upload_dir / filename

    try:
        content = await audio.read()
        if not content:
            raise HTTPException(status_code=400, detail="音频文件为空")

        with upload_path.open("wb") as f:
            f.write(content)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"保存文件失败: {e}")

    # 创建任务记录
    task_info = TaskInfo(
        task_id=task_id,
        status=TaskStatus.PENDING,
        filename=filename,
        created_at=datetime.now(),
        save_segments=save_segments,
    )
    _tasks[task_id] = task_info

    # 添加后台任务
    background_tasks.add_task(
        _process_audio_task,
        request=request,
        task_id=task_id,
        audio_path=upload_path,
        save_segments=save_segments,
        long_segment_threshold_ms=long_segment_threshold_ms,
    )

    logger.info(f"创建任务: {task_id}, 文件: {filename}")

    return CreateTaskResponse(
        task_id=task_id,
        status=TaskStatus.PENDING,
        message="任务已创建，正在处理中...",
    )


@router.get(
    "/status/{task_id}",
    response_model=TaskStatusResponse,
    summary="获取任务状态",
    description="根据任务ID查询处理状态和结果",
)
async def get_task_status(task_id: str) -> TaskStatusResponse:
    """获取任务状态"""
    task = _get_task(task_id)
    return _task_to_response(task)


@router.get(
    "/tasks",
    response_model=TaskListResponse,
    summary="列出所有任务",
    description="获取任务列表，支持按状态过滤",
)
async def list_tasks(
    status: Optional[TaskStatus] = Query(default=None, description="过滤任务状态"),
    limit: int = Query(default=20, ge=1, le=100, description="返回数量限制"),
) -> TaskListResponse:
    """列出所有任务"""
    filtered_tasks = list(_tasks.values())

    if status:
        filtered_tasks = [t for t in filtered_tasks if t.status == status]

    # 按创建时间倒序排列
    sorted_tasks = sorted(filtered_tasks, key=lambda t: t.created_at, reverse=True)[
        :limit
    ]

    return TaskListResponse(
        total=len(_tasks),
        filtered=len(filtered_tasks),
        tasks=[_task_to_response(t) for t in sorted_tasks],
    )


@router.delete(
    "/tasks/{task_id}",
    summary="删除任务",
    description="删除任务及其相关文件",
)
async def delete_task(task_id: str) -> dict:
    """删除任务"""
    # 验证任务存在
    _get_task(task_id)

    # 清理文件
    await _cleanup_task_files(task_id)

    # 删除任务记录
    del _tasks[task_id]

    return {"message": "任务已删除", "task_id": task_id}


# =============================================================================
# API 路由 - 文件下载
# =============================================================================


@router.get(
    "/download/{task_id}/srt",
    summary="下载 SRT 文件",
    description="下载生成的 SRT 字幕文件",
    responses={
        200: {"content": {"text/plain": {}}, "description": "SRT 文件"},
    },
)
async def download_srt(
    task_id: str,
    background_tasks: BackgroundTasks,
) -> FileResponse:
    """下载 SRT 字幕文件"""
    task = _get_task(task_id)

    if task.status != TaskStatus.COMPLETED:
        raise HTTPException(status_code=400, detail="任务尚未完成")

    # 查找 SRT 文件
    srt_filename = Path(task.filename).stem + ".srt"
    srt_path = OUTPUT_DIR / task_id / srt_filename

    if not srt_path.exists():
        raise HTTPException(status_code=404, detail="SRT 文件不存在")

    if CLEANUP_AFTER_DOWNLOAD:
        background_tasks.add_task(_cleanup_task_files, task_id)

    return FileResponse(
        path=srt_path,
        filename=srt_filename,
        media_type="text/plain; charset=utf-8",
    )


@router.get(
    "/download/{task_id}/segments",
    summary="下载音频片段",
    description="下载分割后的音频片段压缩包",
    responses={
        200: {"content": {"application/zip": {}}, "description": "ZIP 压缩包"},
    },
)
async def download_segments(task_id: str) -> FileResponse:
    """下载音频片段压缩包"""
    task = _get_task(task_id)

    if task.status != TaskStatus.COMPLETED:
        raise HTTPException(status_code=400, detail="任务尚未完成")

    if not task.result or not task.result.get("has_segments"):
        raise HTTPException(status_code=404, detail="该任务未保存音频片段")

    segments_dir = OUTPUT_DIR / task_id / "segments"
    if not segments_dir.exists():
        raise HTTPException(status_code=404, detail="音频片段目录不存在")

    # 创建压缩包
    zip_filename = f"{Path(task.filename).stem}_segments.zip"
    zip_path = OUTPUT_DIR / f"{task_id}_segments.zip"

    try:
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
            for file_path in segments_dir.glob("*.wav"):
                zipf.write(file_path, file_path.name)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"创建压缩包失败: {e}")

    return FileResponse(
        path=zip_path,
        filename=zip_filename,
        media_type="application/zip",
    )


@router.get(
    "/content/{task_id}",
    response_model=SRTContentResponse,
    summary="获取 SRT 内容",
    description="直接获取 SRT 文件内容（不下载文件）",
)
async def get_srt_content(task_id: str) -> SRTContentResponse:
    """获取 SRT 内容"""
    task = _get_task(task_id)

    if task.status != TaskStatus.COMPLETED:
        raise HTTPException(status_code=400, detail="任务尚未完成")

    if not task.result or "srt_content" not in task.result:
        raise HTTPException(status_code=404, detail="SRT 内容不可用")

    return SRTContentResponse(
        task_id=task_id,
        srt_content=task.result["srt_content"],
        total_sentences=task.result.get("total_sentences", 0),
    )


# =============================================================================
# API 路由 - 同步处理（小文件快速处理）
# =============================================================================


@router.post(
    "/generate",
    response_class=PlainTextResponse,
    summary="同步生成 SRT（小文件）",
    description="同步处理音频文件，直接返回 SRT 内容。适用于小文件（<30秒）",
    responses={
        200: {"content": {"text/plain": {}}, "description": "SRT 文件内容"},
    },
)
async def generate_srt_sync(
    request: Request,
    audio: UploadFile = File(..., description="音频文件"),
    long_segment_threshold_ms: int = Form(
        default=3000, description="长片段阈值（毫秒）"
    ),
) -> PlainTextResponse:
    """同步生成 SRT（适用于小文件）"""
    _validate_models(request)

    audio_bytes = await audio.read()
    if not audio_bytes:
        raise HTTPException(status_code=400, detail="音频文件为空")

    try:
        config = SRTConfig(long_segment_threshold_ms=long_segment_threshold_ms)
        generator = _get_srt_generator(request, config)

        # 在线程池中执行
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            lambda: generator.process_bytes(audio_bytes),
        )

        # 设置文件名
        filename = audio.filename or "output"
        if filename.endswith((".wav", ".mp3", ".flac", ".m4a", ".ogg")):
            filename = filename.rsplit(".", 1)[0]
        filename = f"{filename}.srt"

        return PlainTextResponse(
            content=result.srt_content,
            media_type="text/plain; charset=utf-8",
            headers={
                "Content-Disposition": f'attachment; filename="{filename}"',
            },
        )

    except Exception as e:
        logger.error(f"SRT 生成失败: {e}")
        raise HTTPException(status_code=500, detail=f"SRT 生成失败: {str(e)}")


# =============================================================================
# API 路由 - 服务状态
# =============================================================================


@router.get(
    "/status",
    response_model=ServiceStatusResponse,
    summary="检查服务状态",
    description="检查 SRT 生成服务状态和模型加载情况",
)
async def check_service_status(request: Request) -> ServiceStatusResponse:
    """检查服务状态"""
    funasr_app = request.app.state.funasr_app
    loaded_models = funasr_app.get_loaded_models()

    # 检查各模型状态
    asr_models = ["paraformer_zh", "sensevoice", "paraformer_en", "whisper"]
    available_asr = [m for m in asr_models if m in loaded_models]

    # 统计任务状态
    pending_count = sum(1 for t in _tasks.values() if t.status == TaskStatus.PENDING)
    processing_count = sum(
        1 for t in _tasks.values() if t.status == TaskStatus.PROCESSING
    )

    return ServiceStatusResponse(
        ready=bool(available_asr) and ("fsmnvad" in loaded_models),
        asr_model=available_asr[0] if available_asr else None,
        vad_model="fsmnvad" in loaded_models,
        punc_model="ctpunc" in loaded_models,
        fazh_model="fazh" in loaded_models,
        loaded_models=loaded_models,
        pending_tasks=pending_count,
        processing_tasks=processing_count,
    )
