import sys
import asyncio
from typing import Dict, Any
from funasr import AutoModel
from pathlib import Path
from src.config import MainConfig, BaseModelConfig
from src.log import logger


# Model name mapping: config field name -> display name for logging
MODEL_DISPLAY_NAMES = {
    "sensevoice": "SenseVoice",
    "paraformer_zh": "Paraformer-ZH",
    "paraformer_zh_streaming": "Paraformer-ZH-Streaming",
    "paraformer_en": "Paraformer-EN",
    "whisper": "Whisper",
    "ctpunc": "CT-Punc",
    "fsmnvad": "FSMN-VAD",
    "fsmnkws": "FSMN-KWS",
    "fazh": "FA-ZH",
    "campplus": "CAM++",
    "emotion2vec": "Emotion2Vec",
    "qwenaudio": "Qwen-Audio",
    "qwenaudio_chat": "Qwen-Audio-Chat",
}


class FunasrApp:
    def __init__(self, config_path: str = "config.toml"):
        self.config_path = config_path
        self.config: MainConfig = None  # type: ignore
        self.models: Dict[str, Any] = {}
        self.lock = asyncio.Lock()

    async def init(self):
        await self.load_config_and_models()

    async def load_config_and_models(self):
        """异步加载配置并初始化所有启用的模型"""
        if sys.version_info >= (3, 11):
            import tomllib
        else:
            import tomli as tomllib
            
        config_path = Path(self.config_path)
        if not config_path.exists():
            logger.error(f"配置文件没有找到: {self.config_path}")
            self.config = MainConfig()
        else:
            async with self.lock:
                try:
                    loop = asyncio.get_event_loop()
                    with open(config_path, "r") as f:
                        content = await loop.run_in_executor(None, f.read)
                    config = tomllib.loads(content)
                    self.config = MainConfig.model_validate(config)
                except Exception as e:
                    logger.error(f"配置文件格式错误: {e}")
                    self.config = MainConfig()
        
        self.models.clear()
        loop = asyncio.get_event_loop()
        
        # 获取所有模型配置字段
        model_config = self.config.model
        
        # 遍历所有模型配置并加载启用的模型
        for field_name in MODEL_DISPLAY_NAMES.keys():
            model_cfg = getattr(model_config, field_name, None)
            if model_cfg is None or not isinstance(model_cfg, BaseModelConfig):
                continue
                
            if not model_cfg.enable:
                logger.info(f"{MODEL_DISPLAY_NAMES[field_name]} 已禁用，跳过加载")
                continue
            
            display_name = MODEL_DISPLAY_NAMES[field_name]
            try:
                logger.info(f"正在加载 {display_name} ({model_cfg.model_name})...")
                
                # 构建设备字符串
                device_str = f"{model_cfg.device}:{model_cfg.device_id}"
                
                # 准备模型参数
                model_kwargs = {
                    "model": model_cfg.model_name,
                    "device": device_str,
                    **(model_cfg.kwargs or {}),
                }
                
                # 如果指定了 model_dir，添加到参数中
                if model_cfg.model_dir:
                    model_kwargs["model_dir"] = model_cfg.model_dir
                
                # 异步加载模型
                self.models[field_name] = await loop.run_in_executor(
                    None,
                    lambda kwargs=model_kwargs: AutoModel(**kwargs)
                )
                logger.info(f"{display_name} 加载成功")
                
            except Exception as e:
                logger.error(f"{display_name} 加载失败: {e}")
        
        loaded_count = len(self.models)
        total_count = len(MODEL_DISPLAY_NAMES)
        logger.info(f"模型加载完成: {loaded_count}/{total_count} 个模型已加载")

    async def get_model(self, name: str):
        """获取指定名称的模型实例"""
        async with self.lock:
            return self.models.get(name)

    def get_model_sync(self, name: str):
        """同步获取指定名称的模型实例（用于非异步上下文）"""
        return self.models.get(name)

    async def reload(self):
        """重新加载配置和模型"""
        async with self.lock:
            await self.load_config_and_models()

    def is_model_loaded(self, name: str) -> bool:
        """检查模型是否已加载"""
        return name in self.models

    def get_loaded_models(self) -> list:
        """获取所有已加载模型的名称列表"""
        return list(self.models.keys())

    async def watch_config(self):
        """监视配置文件变化（待实现）"""
        pass
