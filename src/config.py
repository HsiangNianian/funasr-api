from pydantic import ConfigDict, BaseModel, DirectoryPath, FilePath, Field
from typing import Union, Optional, List, Dict, Any, Tuple, Callable, Literal


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


class CampplusConfig(ConfigModel):
    enable: bool = Field(default=True, description="Enable CampPlus model")
    model_dir: Optional[DirectoryPath] = Field(
        default=None, description="Model directory"
    )
    model_name: str = Field(default="cam++", description="Model name")
    device: Literal["cpu", "cuda"] = Field(default="cuda", description="Device type")
    device_id: Optional[Union[int, str]] = Field(default=0, description="Device ID")
    kwargs: Optional[Dict[str, Any]] = Field(
        default={}, description="Additional keyword arguments for model initialization"
    )


class ConformerConfig(ConfigModel):
    enable: bool = Field(default=True, description="Enable Conformer model")
    model_dir: Optional[DirectoryPath] = Field(
        default=None, description="Model directory"
    )
    model_name: str = Field(default="conformer", description="Model name")
    device: Literal["cpu", "cuda"] = Field(default="cuda", description="Device type")
    device_id: Optional[Union[int, str]] = Field(default=0, description="Device ID")
    kwargs: Optional[Dict[str, Any]] = Field(
        default={}, description="Additional keyword arguments for model initialization"
    )


class CtpuncConfig(ConfigModel):
    enable: bool = Field(default=True, description="Enable Ctpunc model")
    model_dir: Optional[DirectoryPath] = Field(
        default=None, description="Model directory"
    )
    model_name: str = Field(default="conformer", description="Model name")
    device: Literal["cpu", "cuda"] = Field(default="cuda", description="Device type")
    device_id: Optional[Union[int, str]] = Field(default=0, description="Device ID")
    kwargs: Optional[Dict[str, Any]] = Field(
        default={}, description="Additional keyword arguments for model initialization"
    )


class Emotion2VecConfig(ConfigModel):
    enable: bool = Field(default=True, description="Enable Emotion2Vec model")
    model_dir: Optional[DirectoryPath] = Field(
        default=None, description="Model directory"
    )
    model_name: str = Field(default="conformer", description="Model name")
    device: Literal["cpu", "cuda"] = Field(default="cuda", description="Device type")
    device_id: Optional[Union[int, str]] = Field(default=0, description="Device ID")
    kwargs: Optional[Dict[str, Any]] = Field(
        default={}, description="Additional keyword arguments for model initialization"
    )


class FazhConfig(ConfigModel):
    enable: bool = Field(default=True, description="Enable Fazh model")
    model_dir: Optional[DirectoryPath] = Field(
        default=None, description="Model directory"
    )
    model_name: str = Field(default="conformer", description="Model name")
    device: Literal["cpu", "cuda"] = Field(default="cuda", description="Device type")
    device_id: Optional[Union[int, str]] = Field(default=0, description="Device ID")
    kwargs: Optional[Dict[str, Any]] = Field(
        default={}, description="Additional keyword arguments for model initialization"
    )


class FsmnkwsConfig(ConfigModel):
    enable: bool = Field(default=True, description="Enable Fsmnkws model")
    model_dir: Optional[DirectoryPath] = Field(
        default=None, description="Model directory"
    )
    model_name: str = Field(default="conformer", description="Model name")
    device: Literal["cpu", "cuda"] = Field(default="cuda", description="Device type")
    device_id: Optional[Union[int, str]] = Field(default=0, description="Device ID")
    kwargs: Optional[Dict[str, Any]] = Field(
        default={}, description="Additional keyword arguments for model initialization"
    )


class FsmnvadConfig(ConfigModel):
    enable: bool = Field(default=True, description="Enable Fsmnvad model")
    model_dir: Optional[DirectoryPath] = Field(
        default=None, description="Model directory"
    )
    model_name: str = Field(default="conformer", description="Model name")
    device: Literal["cpu", "cuda"] = Field(default="cuda", description="Device type")
    device_id: Optional[Union[int, str]] = Field(default=0, description="Device ID")
    kwargs: Optional[Dict[str, Any]] = Field(
        default={}, description="Additional keyword arguments for model initialization"
    )


class ParaformerConfig(ConfigModel):
    enable: bool = Field(default=True, description="Enable Paraformer model")
    model_dir: Optional[DirectoryPath] = Field(
        default=None, description="Model directory"
    )
    model_name: str = Field(default="conformer", description="Model name")
    device: Literal["cpu", "cuda"] = Field(default="cuda", description="Device type")
    device_id: Optional[Union[int, str]] = Field(default=0, description="Device ID")
    kwargs: Optional[Dict[str, Any]] = Field(
        default={}, description="Additional keyword arguments for model initialization"
    )


class QwenaudioConfig(ConfigModel):
    enable: bool = Field(default=True, description="Enable Qwenaudio model")
    model_dir: Optional[DirectoryPath] = Field(
        default=None, description="Model directory"
    )
    model_name: str = Field(default="conformer", description="Model name")
    device: Literal["cpu", "cuda"] = Field(default="cuda", description="Device type")
    device_id: Optional[Union[int, str]] = Field(default=0, description="Device ID")
    kwargs: Optional[Dict[str, Any]] = Field(
        default={}, description="Additional keyword arguments for model initialization"
    )


class SensevoiceConfig(ConfigModel):
    enable: bool = Field(default=True, description="Enable Sensevoice model")
    model_dir: Optional[DirectoryPath] = Field(
        default=None, description="Model directory"
    )
    model_name: str = Field(default="conformer", description="Model name")
    device: Literal["cpu", "cuda"] = Field(default="cuda", description="Device type")
    device_id: Optional[Union[int, str]] = Field(default=0, description="Device ID")
    kwargs: Optional[Dict[str, Any]] = Field(
        default={}, description="Additional keyword arguments for model initialization"
    )


class WhisperConfig(ConfigModel):
    enable: bool = Field(default=True, description="Enable Whisper model")
    model_dir: Optional[DirectoryPath] = Field(
        default=None, description="Model directory"
    )
    model_name: str = Field(default="conformer", description="Model name")
    device: Literal["cpu", "cuda"] = Field(default="cuda", description="Device type")
    device_id: Optional[Union[int, str]] = Field(default=0, description="Device ID")
    kwargs: Optional[Dict[str, Any]] = Field(
        default={}, description="Additional keyword arguments for model initialization"
    )


class AutoModelConfig(ConfigModel):
    enable: bool = Field(default=True, description="Enable AutoModel")
    model_dir: Optional[DirectoryPath] = Field(
        default=None, description="Model directory"
    )
    model_name: str = Field(default="conformer", description="Model name")
    device: Literal["cpu", "cuda"] = Field(default="cuda", description="Device type")
    device_id: Optional[Union[int, str]] = Field(default=0, description="Device ID")
    kwargs: Optional[Dict[str, Any]] = Field(
        default={}, description="Additional keyword arguments for model initialization"
    )


class ModelConfig(ConfigModel):
    device: Literal["cpu", "cuda"] = Field(default="cuda", description="Device type")
    device_id: Optional[Union[int, str]] = Field(default=0, description="Device ID")
    campplus: "CampplusConfig" = CampplusConfig()
    conformer: "ConformerConfig" = ConformerConfig()
    ctpunc: "CtpuncConfig" = CtpuncConfig()
    emotion2vec: "Emotion2VecConfig" = Emotion2VecConfig()
    fazh: "FazhConfig" = FazhConfig()
    fsmnkws: "FsmnkwsConfig" = FsmnkwsConfig()
    fsmnvad: "FsmnvadConfig" = FsmnvadConfig()
    paraformer: "ParaformerConfig" = ParaformerConfig()
    qwenaudio: "QwenaudioConfig" = QwenaudioConfig()
    sensevoice: "SensevoiceConfig" = SensevoiceConfig()
    whisper: "WhisperConfig" = WhisperConfig()
    auto: "AutoModelConfig" = AutoModelConfig()


class MainConfig(ConfigModel):
    app: AppConfig = AppConfig()
    log: LogConfig = LogConfig()
    model: ModelConfig = ModelConfig()
