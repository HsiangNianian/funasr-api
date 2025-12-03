from pydantic import ConfigDict, BaseModel, Field
from typing import Optional, Union, Dict, Any, Literal


class ConfigModel(BaseModel):
    model_config = ConfigDict(extra="allow")

    __config_name__: str = ""


class LogConfig(ConfigModel):
    level: Union[str, int] = Field(default="DEBUG", description="Logging level")
    verbose_exception: bool = Field(
        default=False, description="Whether to show verbose exception information"
    )


class AppConfig(ConfigModel):
    port: int = Field(default=8000, description="Application port")
    host: str = Field(default="0.0.0.0", description="Application host")


class BaseModelConfig(ConfigModel):
    """Base configuration for all FunASR models"""
    enable: bool = Field(default=True, description="Enable this model")
    model_dir: Optional[str] = Field(default=None, description="Model directory path")
    model_name: str = Field(default="", description="Model name")
    device: Literal["cpu", "cuda"] = Field(default="cuda", description="Device type")
    device_id: Optional[Union[int, str]] = Field(default=0, description="Device ID")
    kwargs: Optional[Dict[str, Any]] = Field(
        default_factory=dict, description="Additional keyword arguments for model initialization"
    )


class SensevoiceConfig(BaseModelConfig):
    """SenseVoice - Multiple speech understanding capabilities (ASR, ITN, LID, SER, AED)"""
    model_name: str = Field(default="iic/SenseVoiceSmall", description="Model name")
    kwargs: Optional[Dict[str, Any]] = Field(
        default_factory=lambda: {
            "vad_model": "fsmn-vad",
            "vad_kwargs": {"max_single_segment_time": 30000},
        },
        description="Additional keyword arguments for model initialization",
    )


class ParaformerZhConfig(BaseModelConfig):
    """Paraformer-zh - Speech recognition with timestamps, non-streaming, Mandarin"""
    model_name: str = Field(default="paraformer-zh", description="Model name")
    kwargs: Optional[Dict[str, Any]] = Field(
        default_factory=lambda: {
            "vad_model": "fsmn-vad",
            "punc_model": "ct-punc",
        },
        description="Additional keyword arguments for model initialization",
    )


class ParaformerZhStreamingConfig(BaseModelConfig):
    """Paraformer-zh-streaming - Speech recognition, streaming, Mandarin"""
    model_name: str = Field(default="paraformer-zh-streaming", description="Model name")


class ParaformerEnConfig(BaseModelConfig):
    """Paraformer-en - Speech recognition without timestamps, non-streaming, English"""
    model_name: str = Field(default="paraformer-en", description="Model name")


class CtpuncConfig(BaseModelConfig):
    """CT-Punc - Punctuation restoration for Mandarin and English"""
    model_name: str = Field(default="ct-punc", description="Model name")


class FsmnvadConfig(BaseModelConfig):
    """FSMN-VAD - Voice activity detection for Mandarin and English"""
    model_name: str = Field(default="fsmn-vad", description="Model name")


class FsmnkwsConfig(BaseModelConfig):
    """FSMN-KWS - Keyword spotting, streaming, Mandarin"""
    model_name: str = Field(default="fsmn-kws", description="Model name")


class FazhConfig(BaseModelConfig):
    """FA-ZH - Timestamp prediction for Mandarin"""
    model_name: str = Field(default="fa-zh", description="Model name")


class CampplusConfig(BaseModelConfig):
    """CAM++ - Speaker verification/diarization"""
    model_name: str = Field(default="cam++", description="Model name")


class WhisperConfig(BaseModelConfig):
    """Whisper-large-v3-turbo - Speech recognition with timestamps, multilingual"""
    model_name: str = Field(default="Whisper-large-v3-turbo", description="Model name")


class QwenaudioConfig(BaseModelConfig):
    """Qwen-Audio - Audio-text multimodal model (pretraining)"""
    model_name: str = Field(default="Qwen-Audio", description="Model name")


class QwenaudioChatConfig(BaseModelConfig):
    """Qwen-Audio-Chat - Audio-text multimodal model (chat)"""
    model_name: str = Field(default="Qwen-Audio-Chat", description="Model name")


class Emotion2VecConfig(BaseModelConfig):
    """Emotion2Vec+large - Speech emotion recognition"""
    model_name: str = Field(default="iic/emotion2vec_plus_large", description="Model name")


class ModelConfig(ConfigModel):
    """Configuration for all FunASR models"""
    device: Literal["cpu", "cuda"] = Field(default="cuda", description="Default device type")
    device_id: Optional[Union[int, str]] = Field(default=0, description="Default device ID")
    
    # Speech Recognition Models
    sensevoice: SensevoiceConfig = Field(default_factory=SensevoiceConfig)
    paraformer_zh: ParaformerZhConfig = Field(default_factory=ParaformerZhConfig)
    paraformer_zh_streaming: ParaformerZhStreamingConfig = Field(default_factory=ParaformerZhStreamingConfig)
    paraformer_en: ParaformerEnConfig = Field(default_factory=ParaformerEnConfig)
    whisper: WhisperConfig = Field(default_factory=WhisperConfig)
    
    # Punctuation & VAD & Timestamp
    ctpunc: CtpuncConfig = Field(default_factory=CtpuncConfig)
    fsmnvad: FsmnvadConfig = Field(default_factory=FsmnvadConfig)
    fsmnkws: FsmnkwsConfig = Field(default_factory=FsmnkwsConfig)
    fazh: FazhConfig = Field(default_factory=FazhConfig)
    
    # Speaker & Emotion
    campplus: CampplusConfig = Field(default_factory=CampplusConfig)
    emotion2vec: Emotion2VecConfig = Field(default_factory=Emotion2VecConfig)
    
    # Multimodal
    qwenaudio: QwenaudioConfig = Field(default_factory=QwenaudioConfig)
    qwenaudio_chat: QwenaudioChatConfig = Field(default_factory=QwenaudioChatConfig)


class MainConfig(ConfigModel):
    app: AppConfig = Field(default_factory=AppConfig)
    log: LogConfig = Field(default_factory=LogConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
