from re import L
import sys
import asyncio
from typing import Dict, Any
from funasr import AutoModel
from pathlib import Path
from src.config import MainConfig
from src.log import logger

class FunasrApp:
    def __init__(self, config_path: str = "config.toml"):
        self.config_path = config_path
        self.config = None
        self.models: Dict[str, Any] = {}
        self.lock = asyncio.Lock()

    async def init(self):
        await self.load_config_and_models()

    async def load_config_and_models(self):
        """异步加载配置并初始化所有模型"""
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
        try:
            loop = asyncio.get_event_loop()
            sensevoice_cfg = self.config.model.sensevoice
            self.models["sensevoice"] = await loop.run_in_executor(
                None,
                lambda: AutoModel(
                    model=sensevoice_cfg.model_name,
                    vad_model="fsmn-vad",
                    vad_kwargs=sensevoice_cfg.kwargs or {"max_single_segment_time": 30000},
                    device=f"{sensevoice_cfg.device}:{sensevoice_cfg.device_id}",
                )
            )
        except Exception as e:
            logger.error(f"SenseVoice 加载失败: {e}")
        try:

            loop = asyncio.get_event_loop()
            whisper_cfg = self.config.model.whisper
            self.models["whisper"] = await loop.run_in_executor(
                None,
                lambda: AutoModel(
                    model=whisper_cfg.model_name,
                    device=f"{whisper_cfg.device}:{whisper_cfg.device_id}",
                    **(whisper_cfg.kwargs or {})
                )
            )
        except Exception as e:
            logger.error(f"Whisper 加载失败: {e}")
        try:
            loop = asyncio.get_event_loop()
            auto_cfg = self.config.model.auto
            self.models["auto"] = await loop.run_in_executor(
                None,
                lambda: AutoModel(
                    model=auto_cfg.model_name,
                    device=f"{auto_cfg.device}:{auto_cfg.device_id}",
                    **(auto_cfg.kwargs or {})
                )
            )
        except Exception as e:
            logger.error(f"AutoModel 加载失败: {e}")
        # try:
        #     loop = asyncio.get_event_loop()
        #     conformer_cfg = self.config.model.conformer
        #     self.models["conformer"] = await loop.run_in_executor(
        #         None,
        #         lambda: AutoModel(
        #             model=conformer_cfg.model_name,
        #             device=f"{conformer_cfg.device}:{conformer_cfg.device_id}",
        #             **(conformer_cfg.kwargs or {})
        #         )
        #     )
        # except Exception as e:
        #     logger.error(f"Conformer 加载失败: {e}")
        try:
            loop = asyncio.get_event_loop()
            campplus_cfg = self.config.model.campplus
            self.models["campplus"] = await loop.run_in_executor(
                None,
                lambda: AutoModel(
                    model=campplus_cfg.model_name,
                    device=f"{campplus_cfg.device}:{campplus_cfg.device_id}",
                    **(campplus_cfg.kwargs or {})
                )
            )
        except Exception as e:
            logger.error(f"CampPlus 加载失败: {e}")
        try:
            loop = asyncio.get_event_loop()
            paraformer_cfg = self.config.model.paraformer
            self.models["paraformer"] = await loop.run_in_executor(
                None,
                lambda: AutoModel(
                    model=paraformer_cfg.model_name,
                    device=f"{paraformer_cfg.device}:{paraformer_cfg.device_id}",
                    **(paraformer_cfg.kwargs or {})
                )
            )
        except Exception as e:
            logger.error(f"Paraformer 加载失败: {e}")
        try:
            loop = asyncio.get_event_loop()
            fsmnvad_cfg = self.config.model.fsmnvad
            self.models["fsmnvad"] = await loop.run_in_executor(
                None,
                lambda: AutoModel(
                    model=fsmnvad_cfg.model_name,
                    device=f"{fsmnvad_cfg.device}:{fsmnvad_cfg.device_id}",
                    **(fsmnvad_cfg.kwargs or {})
                )
            )
        except Exception as e:
            logger.error(f"FsmnVAD 加载失败: {e}")
        try:
            loop = asyncio.get_event_loop()
            fsmnkws_cfg = self.config.model.fsmnkws
            self.models["fsmnkws"] = await loop.run_in_executor(
                None,
                lambda: AutoModel(
                    model=fsmnkws_cfg.model_name,
                    device=f"{fsmnkws_cfg.device}:{fsmnkws_cfg.device_id}",
                    **(fsmnkws_cfg.kwargs or {})
                )
            )
        except Exception as e:
            logger.error(f"FsmnKWS 加载失败: {e}")
        try:
            loop = asyncio.get_event_loop()
            ctpunc_cfg = self.config.model.ctpunc
            self.models["ctpunc"] = await loop.run_in_executor(
                None,
                lambda: AutoModel(
                    model=ctpunc_cfg.model_name,
                    device=f"{ctpunc_cfg.device}:{ctpunc_cfg.device_id}",
                    **(ctpunc_cfg.kwargs or {})
                )
            )
        except Exception as e:
            logger.error(f"CTPunc 加载失败: {e}")
        try:
            loop = asyncio.get_event_loop()
            emotion2vec_cfg = self.config.model.emotion2vec
            self.models["emotion2vec"] = await loop.run_in_executor(
                None,
                lambda: AutoModel(
                    model=emotion2vec_cfg.model_name,
                    device=f"{emotion2vec_cfg.device}:{emotion2vec_cfg.device_id}",
                    **(emotion2vec_cfg.kwargs or {})
                )
            )
        except Exception as e:
            logger.error(f"Emotion2Vec 加载失败: {e}")
        try:
            loop = asyncio.get_event_loop()
            fazh_cfg = self.config.model.fazh
            self.models["fazh"] = await loop.run_in_executor(
                None,
                lambda: AutoModel(
                    model=fazh_cfg.model_name,
                    device=f"{fazh_cfg.device}:{fazh_cfg.device_id}",
                    **(fazh_cfg.kwargs or {})
                )
            )
        except Exception as e:
            logger.error(f"FaZh 加载失败: {e}")
        try:
            loop = asyncio.get_event_loop()
            qwenaudio_cfg = self.config.model.qwenaudio
            self.models["qwenaudio"] = await loop.run_in_executor(
                None,
                lambda: AutoModel(
                    model=qwenaudio_cfg.model_name,
                    device=f"{qwenaudio_cfg.device}:{qwenaudio_cfg.device_id}",
                    **(qwenaudio_cfg.kwargs or {})
                )
            )
        except Exception as e:
            logger.error(f"QwenAudio 加载失败: {e}")
            
        logger.info("所有模型加载完成。")    
        
        
    async def get_model(self, name: str):
        async with self.lock:
            return self.models.get(name)

    async def reload(self):
        async with self.lock:
            await self.load_config_and_models()

    async def watch_config(self):
        pass
