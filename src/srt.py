"""
SRT 字幕生成模块

基于 FunASR 模型生成高精度 SRT 字幕文件
支持 VAD 分段、ASR 识别、标点恢复、时间戳对齐
"""

from __future__ import annotations

import asyncio
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

import numpy as np
import soundfile as sf

from src.log import logger

if TYPE_CHECKING:
    from src.app import FunasrApp

# =============================================================================
# 类型别名定义
# =============================================================================

TimestampMs = int  # 以毫秒为单位的时间戳
AudioSegment = np.ndarray  # 音频采样数据
VadSegment = Tuple[TimestampMs, TimestampMs]  # VAD片段 (start_ms, end_ms)
WordTimestamp = Tuple[str, List[TimestampMs]]  # 词时间戳 (word, [start_ms, end_ms])
SentenceTimestamp = Tuple[str, List[TimestampMs]]  # 句子时间戳 (sentence, [start_ms, end_ms])


# =============================================================================
# 数据类定义
# =============================================================================


@dataclass
class SRTConfig:
    """SRT 生成配置"""

    # 句子分隔符
    sentence_delimiters: List[str] = field(
        default_factory=lambda: [
            "。", "！", "？", "；", "，", "、", "：",
            """, """, "'", "'", '"', "'",
        ]
    )
    # 长片段阈值（毫秒），超过此值使用 FA-ZH 细化时间戳
    long_segment_threshold_ms: int = 3000
    # 目标采样率
    target_sample_rate: int = 16000
    # 是否保存音频片段
    save_segments: bool = False
    # 无效文本列表（跳过这些文本）
    invalid_texts: List[str] = field(default_factory=list)


@dataclass
class ProcessingResult:
    """处理结果"""

    segments: List[SentenceTimestamp]  # 句子级时间戳
    vad_segments: List[VadSegment]  # VAD 分段
    char_timestamps: List[WordTimestamp]  # 字符级时间戳
    srt_content: str  # SRT 文件内容
    metadata: Dict[str, Any] = field(default_factory=dict)


# =============================================================================
# 音频处理工具类
# =============================================================================


class AudioProcessor:
    """音频处理工具类"""

    def __init__(self, sample_rate: int = 16000):
        self.sample_rate = sample_rate

    def ms_to_samples(self, ms: TimestampMs) -> int:
        """毫秒转换为采样点"""
        return int(ms * self.sample_rate / 1000)

    def samples_to_ms(self, samples: int) -> TimestampMs:
        """采样点转换为毫秒"""
        return int(samples * 1000 / self.sample_rate)

    def extract_segment(
        self, audio_data: AudioSegment, start_ms: TimestampMs, end_ms: TimestampMs
    ) -> AudioSegment:
        """根据时间戳提取音频片段"""
        start_sample = self.ms_to_samples(start_ms)
        end_sample = self.ms_to_samples(end_ms)
        return audio_data[start_sample:end_sample]

    def get_duration_ms(self, audio_data: AudioSegment) -> TimestampMs:
        """获取音频时长（毫秒）"""
        return self.samples_to_ms(len(audio_data))

    @staticmethod
    def load_audio(audio_path: Path) -> Tuple[AudioSegment, int]:
        """加载音频文件，返回音频采样数据和采样率"""
        if not audio_path.exists():
            raise FileNotFoundError(f"音频文件不存在: {audio_path}")
        audio_data, sample_rate = sf.read(str(audio_path))
        # 如果是立体声，转换为单声道
        if len(audio_data.shape) > 1:
            audio_data = np.mean(audio_data, axis=1)
        return audio_data.astype(np.float32), sample_rate

    @staticmethod
    def load_audio_bytes(audio_bytes: bytes) -> Tuple[AudioSegment, int]:
        """从字节流加载音频"""
        import io
        audio_data, sample_rate = sf.read(io.BytesIO(audio_bytes))
        if len(audio_data.shape) > 1:
            audio_data = np.mean(audio_data, axis=1)
        return audio_data.astype(np.float32), sample_rate

    @staticmethod
    def save_audio(
        audio_data: AudioSegment, output_path: Path, sample_rate: int = 16000
    ) -> None:
        """保存音频文件"""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(output_path), audio_data, samplerate=sample_rate)

    @staticmethod
    def resample_audio(
        audio_data: AudioSegment, original_rate: int, target_rate: int
    ) -> AudioSegment:
        """重采样音频到目标采样率"""
        if original_rate == target_rate:
            return audio_data
        try:
            import librosa
            return librosa.resample(
                audio_data, orig_sr=original_rate, target_sr=target_rate
            )
        except ImportError:
            logger.warning("librosa 未安装，无法重采样音频")
            return audio_data


# =============================================================================
# SRT 生成器
# =============================================================================


class SRTGenerator:
    """
    SRT 字幕生成器
    
    使用 FunasrApp 中已加载的模型生成高精度 SRT 字幕
    
    所需模型:
    - paraformer_zh 或其他 ASR 模型: 语音识别
    - fsmnvad: VAD 语音活动检测
    - ctpunc: 标点恢复
    - fazh: 时间戳预测（可选，用于长片段细化）
    """

    def __init__(
        self,
        funasr_app: "FunasrApp",
        config: Optional[SRTConfig] = None,
    ):
        self.app = funasr_app
        self.config = config or SRTConfig()
        self.audio_processor = AudioProcessor(self.config.target_sample_rate)

    # -------------------------------------------------------------------------
    # 模型访问
    # -------------------------------------------------------------------------

    def _get_asr_model(self) -> Any:
        """获取 ASR 模型（优先使用 paraformer_zh）"""
        for model_name in ["paraformer_zh", "sensevoice", "paraformer_en", "whisper"]:
            model = self.app.get_model_sync(model_name)
            if model is not None:
                return model
        raise RuntimeError("没有可用的 ASR 模型，请在配置中启用 paraformer_zh 或其他 ASR 模型")

    def _get_vad_model(self) -> Any:
        """获取 VAD 模型"""
        model = self.app.get_model_sync("fsmnvad")
        if model is None:
            raise RuntimeError("VAD 模型未加载，请在配置中启用 fsmnvad")
        return model

    def _get_punc_model(self) -> Optional[Any]:
        """获取标点模型（可选）"""
        return self.app.get_model_sync("ctpunc")

    def _get_fazh_model(self) -> Optional[Any]:
        """获取时间戳预测模型（可选）"""
        return self.app.get_model_sync("fazh")

    # -------------------------------------------------------------------------
    # 核心处理方法
    # -------------------------------------------------------------------------

    def get_vad_segments(
        self, audio_data: AudioSegment, sample_rate: Optional[int] = None
    ) -> List[VadSegment]:
        """获取 VAD 分段结果"""
        sample_rate = sample_rate or self.audio_processor.sample_rate
        vad_model = self._get_vad_model()
        
        vad_result = vad_model.generate(input=audio_data, sample_rate=sample_rate)
        if not vad_result or "value" not in vad_result[0]:
            return []
        return vad_result[0]["value"]

    def get_asr_text(self, audio_segment: AudioSegment) -> str:
        """获取 ASR 识别文本（会先用 VAD 分段）"""
        text = ""
        vad_segments = self.get_vad_segments(
            audio_segment, sample_rate=self.audio_processor.sample_rate
        )
        if len(vad_segments) == 0:
            return text
        for vad_segment in vad_segments:
            start_ms, end_ms = vad_segment
            segment_audio = self.audio_processor.extract_segment(
                audio_segment, start_ms, end_ms
            )
            text += self._get_asr_text(segment_audio)
        return text

    def _get_asr_text(self, audio_segment: AudioSegment) -> str:
        """内部方法：直接获取 ASR 识别文本"""
        asr_model = self._get_asr_model()
        result = asr_model.generate(
            input=audio_segment, sample_rate=self.audio_processor.sample_rate
        )
        if not result or "text" not in result[0]:
            return ""
        # 移除空格
        return "".join(result[0]["text"].split())

    def get_punc_text(self, text: str) -> str:
        """获取带标点的文本"""
        punc_model = self._get_punc_model()
        if punc_model is None:
            return text  # 无标点模型，返回原文本
        result = punc_model.generate(input=text)
        if not result or "text" not in result[0]:
            return text
        return result[0]["text"]

    def get_fazh_result(
        self, audio_path: Path, text_path: Path
    ) -> List[Dict[str, Any]]:
        """获取 FA-ZH 时间戳预测结果"""
        fazh_model = self._get_fazh_model()
        if fazh_model is None:
            raise RuntimeError("FA-ZH 模型未加载")
        
        result = fazh_model.generate(
            input=(str(audio_path), str(text_path)), data_type=("sound", "text")
        )
        return result

    # -------------------------------------------------------------------------
    # 文本处理方法
    # -------------------------------------------------------------------------

    def serialize_sentences(self, text: str) -> List[str]:
        """将带标点的文本切分为句子列表"""
        result = text
        for delimiter in self.config.sentence_delimiters:
            result = result.replace(delimiter, "|")
        sentences = [s.strip() for s in result.split("|") if s.strip()]
        return sentences

    def remove_punctuation(self, text: str) -> str:
        """移除标点符号"""
        result = text
        for delimiter in self.config.sentence_delimiters:
            result = result.replace(delimiter, "")
        return result

    def is_valid_text(self, text: str) -> bool:
        """检查文本是否有效"""
        return len(text) > 0 and text not in self.config.invalid_texts

    # -------------------------------------------------------------------------
    # 时间戳处理方法
    # -------------------------------------------------------------------------

    def fix_word_timestamps(
        self,
        word_timestamps: List[WordTimestamp],
        vad_segment: VadSegment,
    ) -> List[WordTimestamp]:
        """修正 FA-ZH 返回的单词时间戳"""
        if not word_timestamps:
            return []

        vad_start, vad_end = vad_segment

        # 计算首尾时间差异的平均调整量
        first_word_start_diff = word_timestamps[0][1][0] - vad_start
        last_word_end_diff = word_timestamps[-1][1][1] - vad_end
        avg_adjustment = (first_word_start_diff + last_word_end_diff) / 2 + 10

        fixed_timestamps: List[WordTimestamp] = []
        for i, (word, timestamp) in enumerate(word_timestamps):
            start_time, end_time = timestamp

            if i == 0:
                fixed_timestamps.append(
                    (word, [vad_start, int(end_time - avg_adjustment)])
                )
            elif i == len(word_timestamps) - 1:
                fixed_timestamps.append(
                    (word, [int(start_time - avg_adjustment), vad_end])
                )
            else:
                fixed_timestamps.append(
                    (
                        word,
                        [
                            int(start_time - avg_adjustment),
                            int(end_time - avg_adjustment),
                        ],
                    )
                )

        return fixed_timestamps

    def align_sentence_timestamps(
        self,
        word_timestamps: List[WordTimestamp],
        sentences: List[str],
    ) -> List[SentenceTimestamp]:
        """对齐句子时间戳"""
        total_chars = len("".join(sentences))
        if len(word_timestamps) != total_chars:
            logger.warning(
                f"单词时间戳数量({len(word_timestamps)})与句子总字符数({total_chars})不匹配"
            )
            # 尝试容错处理
            if len(word_timestamps) == 0:
                return []

        aligned_timestamps: List[SentenceTimestamp] = []
        char_index = 0

        for sentence in sentences:
            sentence_len = len(sentence)
            if sentence_len == 0:
                continue

            if char_index >= len(word_timestamps):
                break

            # 获取句子首字符和末字符的时间戳
            start_time = word_timestamps[char_index][1][0]
            end_idx = min(char_index + sentence_len - 1, len(word_timestamps) - 1)
            end_time = word_timestamps[end_idx][1][1]

            aligned_timestamps.append((sentence, [start_time, end_time]))
            char_index += sentence_len

        return aligned_timestamps

    def _extract_char_timestamps_from_sentences(
        self, sentence_timestamps: List[SentenceTimestamp]
    ) -> List[WordTimestamp]:
        """从句子时间戳中提取字符级时间戳"""
        char_timestamps: List[WordTimestamp] = []

        for sentence, timestamp in sentence_timestamps:
            start_ms, end_ms = timestamp
            sentence_len = len(sentence)

            if sentence_len == 0:
                continue

            duration_per_char = (end_ms - start_ms) / sentence_len

            for i, char in enumerate(sentence):
                char_start = int(start_ms + i * duration_per_char)
                char_end = int(start_ms + (i + 1) * duration_per_char)
                char_timestamps.append((char, [char_start, char_end]))

        return char_timestamps

    def _generate_uniform_char_timestamps(
        self, text: str, start_ms: TimestampMs, end_ms: TimestampMs
    ) -> List[WordTimestamp]:
        """为短片段生成平均分配的字符时间戳"""
        char_timestamps: List[WordTimestamp] = []
        text_len = len(text)

        if text_len == 0:
            return char_timestamps

        duration_per_char = (end_ms - start_ms) / text_len

        for i, char in enumerate(text):
            char_start = int(start_ms + i * duration_per_char)
            char_end = int(start_ms + (i + 1) * duration_per_char)
            char_timestamps.append((char, [char_start, char_end]))

        return char_timestamps

    # -------------------------------------------------------------------------
    # 片段处理方法
    # -------------------------------------------------------------------------

    def process_vad_segment(
        self,
        audio_data: AudioSegment,
        vad_segment: VadSegment,
    ) -> str:
        """处理单个 VAD 片段，返回带标点的文本"""
        start_ms, end_ms = vad_segment
        segment_audio = self.audio_processor.extract_segment(
            audio_data, start_ms, end_ms
        )
        
        asr_text = self.get_asr_text(segment_audio)
        if len(asr_text) == 0:
            return ""
        
        punc_text = self.get_punc_text(asr_text)
        return punc_text

    def process_long_segment(
        self,
        audio_data: AudioSegment,
        vad_segment: VadSegment,
        punc_text: str,
        temp_dir: Path,
    ) -> List[SentenceTimestamp]:
        """处理长音频片段，使用 FA-ZH 进行细粒度对齐"""
        start_ms, end_ms = vad_segment

        # 保存临时音频文件
        temp_audio_path = temp_dir / f"temp_fazh_{start_ms}_{end_ms}.wav"
        segment_audio = self.audio_processor.extract_segment(
            audio_data, start_ms, end_ms
        )
        self.audio_processor.save_audio(
            segment_audio, temp_audio_path, self.audio_processor.sample_rate
        )

        # 保存临时文本文件（移除标点）
        temp_text_path = temp_dir / f"temp_fazh_{start_ms}_{end_ms}.txt"
        plain_text = self.remove_punctuation(punc_text)
        temp_text_path.write_text(plain_text, encoding="utf-8")

        try:
            # 获取 FA-ZH 结果
            fazh_result = self.get_fazh_result(temp_audio_path, temp_text_path)
            
            if not fazh_result:
                raise ValueError("FA-ZH 返回结果为空")
            
            fazh_text = fazh_result[0]["text"].split()
            fazh_timestamps = fazh_result[0]["timestamp"]

            # FA-ZH 的时间戳是相对于片段的，需要转换为绝对时间戳
            # 同时展开多字符的 token（如 "VY"）
            absolute_timestamps: List[WordTimestamp] = []

            for token, timestamp_pair in zip(fazh_text, fazh_timestamps):
                absolute_start = timestamp_pair[0] + start_ms
                absolute_end = timestamp_pair[1] + start_ms

                if len(token) > 1:
                    # 多字符 token，平均分配时间戳给每个字符
                    duration_per_char = (absolute_end - absolute_start) / len(token)
                    for i, char in enumerate(token):
                        char_start = int(absolute_start + i * duration_per_char)
                        char_end = int(absolute_start + (i + 1) * duration_per_char)
                        absolute_timestamps.append((char, [char_start, char_end]))
                else:
                    absolute_timestamps.append((token, [absolute_start, absolute_end]))

            # 修正时间戳
            fixed_timestamps = self.fix_word_timestamps(absolute_timestamps, vad_segment)

            # 强制对齐到句子
            sentences = self.serialize_sentences(punc_text)
            aligned_timestamps = self.align_sentence_timestamps(
                fixed_timestamps, sentences
            )

            return aligned_timestamps

        except Exception as e:
            logger.warning(f"FA-ZH 处理失败，回退到简单处理: {e}")
            # 回退到简单处理：将所有句子均匀分配时间
            sentences = self.serialize_sentences(punc_text)
            if not sentences:
                return []
            
            total_duration = end_ms - start_ms
            duration_per_sentence = total_duration / len(sentences)
            result_timestamps: List[SentenceTimestamp] = []
            
            for i, sentence in enumerate(sentences):
                sent_start = int(start_ms + i * duration_per_sentence)
                sent_end = int(start_ms + (i + 1) * duration_per_sentence)
                result_timestamps.append((sentence, [sent_start, sent_end]))
            
            return result_timestamps

        finally:
            # 清理临时文件
            try:
                temp_audio_path.unlink(missing_ok=True)
                temp_text_path.unlink(missing_ok=True)
            except Exception:
                pass

    # -------------------------------------------------------------------------
    # SRT 格式化方法
    # -------------------------------------------------------------------------

    @staticmethod
    def ms_to_srt_time(ms: TimestampMs) -> str:
        """毫秒转换为 SRT 时间格式 (HH:MM:SS,mmm)"""
        total_ms = int(round(ms))
        hours = total_ms // 3600000
        remaining = total_ms % 3600000
        minutes = remaining // 60000
        remaining = remaining % 60000
        seconds = remaining // 1000
        milliseconds = remaining % 1000
        return f"{hours:02d}:{minutes:02d}:{seconds:02d},{milliseconds:03d}"

    def generate_srt_content(
        self, sentence_timestamps: List[SentenceTimestamp]
    ) -> str:
        """生成 SRT 文件内容"""
        lines: List[str] = []
        for idx, (text, timestamp) in enumerate(sentence_timestamps, 1):
            start_time_str = self.ms_to_srt_time(timestamp[0])
            end_time_str = self.ms_to_srt_time(timestamp[1])
            lines.append(f"{idx}")
            lines.append(f"{start_time_str} --> {end_time_str}")
            lines.append(text)
            lines.append("")  # 空行分隔
        return "\n".join(lines)

    def save_srt_file(
        self,
        sentence_timestamps: List[SentenceTimestamp],
        output_path: Path,
    ) -> None:
        """保存 SRT 字幕文件"""
        content = self.generate_srt_content(sentence_timestamps)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(content, encoding="utf-8")
        logger.info(f"SRT 文件已保存: {output_path}")

    # -------------------------------------------------------------------------
    # 主处理方法
    # -------------------------------------------------------------------------

    def process(
        self,
        audio_data: AudioSegment,
        sample_rate: int,
        output_dir: Optional[Path] = None,
    ) -> ProcessingResult:
        """
        处理音频数据，生成 SRT 字幕
        
        Args:
            audio_data: 音频采样数据
            sample_rate: 采样率
            output_dir: 输出目录（可选，用于保存临时文件和结果）
        
        Returns:
            ProcessingResult: 处理结果
        """
        # 重采样到目标采样率
        if sample_rate != self.audio_processor.sample_rate:
            audio_data = self.audio_processor.resample_audio(
                audio_data,
                original_rate=sample_rate,
                target_rate=self.audio_processor.sample_rate,
            )
            sample_rate = self.audio_processor.sample_rate

        # 获取 VAD 分段
        vad_segments = self.get_vad_segments(audio_data, sample_rate=sample_rate)
        logger.info(f"VAD 分段数量: {len(vad_segments)}")

        # 处理每个 VAD 片段
        vad_punc_texts: List[str] = []
        for vad_segment in vad_segments:
            punc_text = self.process_vad_segment(audio_data, vad_segment)
            vad_punc_texts.append(punc_text)

        all_sentence_timestamps: List[SentenceTimestamp] = []
        all_char_timestamps: List[WordTimestamp] = []

        # 使用临时目录或指定的输出目录
        with tempfile.TemporaryDirectory() as temp_dir_str:
            temp_dir = Path(temp_dir_str) if output_dir is None else output_dir

            for vad_segment, punc_text in zip(vad_segments, vad_punc_texts):
                if not self.is_valid_text(punc_text):
                    continue

                start_ms, end_ms = vad_segment
                duration_ms = end_ms - start_ms

                # 检查是否有 FA-ZH 模型可用
                fazh_model = self._get_fazh_model()

                # 长片段使用 FA-ZH 细化（如果可用）
                if duration_ms > self.config.long_segment_threshold_ms and fazh_model:
                    sentence_timestamps = self.process_long_segment(
                        audio_data, vad_segment, punc_text, temp_dir
                    )
                    all_sentence_timestamps.extend(sentence_timestamps)

                    char_timestamps = self._extract_char_timestamps_from_sentences(
                        sentence_timestamps
                    )
                    all_char_timestamps.extend(char_timestamps)
                else:
                    # 短片段或无 FA-ZH 模型，直接使用 VAD 时间戳
                    sentences = self.serialize_sentences(punc_text)
                    if sentences:
                        # 如果有多个句子，平均分配时间
                        total_duration = end_ms - start_ms
                        duration_per_sentence = total_duration / len(sentences)
                        
                        for i, sentence in enumerate(sentences):
                            sent_start = int(start_ms + i * duration_per_sentence)
                            sent_end = int(start_ms + (i + 1) * duration_per_sentence)
                            all_sentence_timestamps.append((sentence, [sent_start, sent_end]))
                            
                            char_timestamps = self._generate_uniform_char_timestamps(
                                sentence, sent_start, sent_end
                            )
                            all_char_timestamps.extend(char_timestamps)

        # 生成 SRT 内容
        srt_content = self.generate_srt_content(all_sentence_timestamps)

        logger.info(f"生成 {len(all_sentence_timestamps)} 条字幕")

        return ProcessingResult(
            segments=all_sentence_timestamps,
            vad_segments=vad_segments,
            char_timestamps=all_char_timestamps,
            srt_content=srt_content,
            metadata={
                "total_sentences": len(all_sentence_timestamps),
                "total_vad_segments": len(vad_segments),
                "audio_duration_ms": self.audio_processor.get_duration_ms(audio_data),
                "sample_rate": sample_rate,
            },
        )

    def process_file(
        self,
        audio_path: Path,
        output_dir: Optional[Path] = None,
        save_srt: bool = True,
        save_segments: bool = False,
    ) -> ProcessingResult:
        """
        处理音频文件，生成 SRT 字幕
        
        Args:
            audio_path: 音频文件路径
            output_dir: 输出目录
            save_srt: 是否保存 SRT 文件
            save_segments: 是否保存音频片段
        
        Returns:
            ProcessingResult: 处理结果
        """
        logger.info(f"处理音频文件: {audio_path}")

        # 加载音频
        audio_data, sample_rate = self.audio_processor.load_audio(audio_path)
        logger.info(f"音频采样率: {sample_rate}, 时长: {len(audio_data)/sample_rate:.2f}s")

        # 处理音频
        result = self.process(audio_data, sample_rate, output_dir)

        # 保存结果
        if output_dir and save_srt:
            output_dir.mkdir(parents=True, exist_ok=True)
            srt_path = output_dir / f"{audio_path.stem}.srt"
            self.save_srt_file(result.segments, srt_path)

        if output_dir and save_segments:
            self._save_audio_segments(
                output_dir, audio_data, result.segments
            )

        return result

    async def process_file_async(
        self,
        audio_path: Path,
        output_dir: Optional[Path] = None,
        save_srt: bool = True,
        save_segments: bool = False,
    ) -> ProcessingResult:
        """异步处理音频文件"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            lambda: self.process_file(audio_path, output_dir, save_srt, save_segments),
        )

    def process_bytes(
        self,
        audio_bytes: bytes,
        output_dir: Optional[Path] = None,
    ) -> ProcessingResult:
        """
        处理音频字节流，生成 SRT 字幕
        
        Args:
            audio_bytes: 音频字节数据
            output_dir: 输出目录（可选）
        
        Returns:
            ProcessingResult: 处理结果
        """
        audio_data, sample_rate = self.audio_processor.load_audio_bytes(audio_bytes)
        return self.process(audio_data, sample_rate, output_dir)

    async def process_bytes_async(
        self,
        audio_bytes: bytes,
        output_dir: Optional[Path] = None,
    ) -> ProcessingResult:
        """异步处理音频字节流"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            lambda: self.process_bytes(audio_bytes, output_dir),
        )

    def _save_audio_segments(
        self,
        output_dir: Path,
        audio_data: AudioSegment,
        sentence_timestamps: List[SentenceTimestamp],
    ) -> None:
        """保存音频片段"""
        segments_dir = output_dir / "segments"
        segments_dir.mkdir(exist_ok=True)

        for idx, (text, timestamp) in enumerate(sentence_timestamps, 1):
            start_ms, end_ms = timestamp
            segment = self.audio_processor.extract_segment(audio_data, start_ms, end_ms)

            # 生成安全的文件名
            safe_text = text[:20].replace(" ", "_").replace("/", "_").replace("\\", "_")
            for char in ['<', '>', ':', '"', '|', '?', '*']:
                safe_text = safe_text.replace(char, "_")
            
            output_path = segments_dir / f"segment_{idx:04d}_{safe_text}.wav"
            self.audio_processor.save_audio(
                segment, output_path, self.audio_processor.sample_rate
            )

        logger.info(f"音频片段已保存到: {segments_dir}")
