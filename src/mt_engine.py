"""
机器翻译引擎模块 — MTEngine
基于 Helsinki-NLP/opus-mt-en-zh，将英文文本翻译为中文
支持上下文增强翻译、模型本地加载、降级策略
"""
import os
import time
from pathlib import Path
from typing import List, Optional

# ─── 强制模型缓存落在项目目录 ───
PROJECT_ROOT = Path(__file__).parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
os.environ.setdefault("HF_HOME", str(MODELS_DIR / ".hf_cache"))
os.environ.setdefault("TRANSFORMERS_CACHE", str(MODELS_DIR))

# ─── 上下文窗口大小 ───
CONTEXT_SIZE = 3  # 翻译当前句时拼接前 N 句英文原文作为上下文


class MTEngine:
    """
    英→中 翻译引擎

    用法:
        mt = MTEngine()
        mt.load_model()
        result = mt.translate("Hello world.")
        # 带上下文
        result = mt.translate("It is fast.", context=["Apple released a chip."])
    """

    def __init__(
        self,
        source_lang: str = "en",
        target_lang: str = "zh",
    ):
        self.source_lang = source_lang
        self.target_lang = target_lang

        self._model = None
        self._tokenizer = None
        self._load_time: float = 0.0

    def load_model(self) -> float:
        """加载翻译模型，返回耗时 (秒)"""
        from transformers import MarianMTModel, MarianTokenizer

        t0 = time.time()

        # 优先使用本地模型
        local_mt = MODELS_DIR / "opus-mt-en-zh"
        if local_mt.exists() and any(local_mt.iterdir()):
            model_name = str(local_mt)
        else:
            model_name = "Helsinki-NLP/opus-mt-en-zh"

        print(f"[MT] 加载模型: {model_name} ...")
        self._tokenizer = MarianTokenizer.from_pretrained(model_name)
        self._model = MarianMTModel.from_pretrained(model_name)

        self._load_time = time.time() - t0
        return self._load_time

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    def translate(self, en_text: str, context: Optional[List[str]] = None) -> str:
        """
        翻译英文文本为中文

        Args:
            en_text: 英文原文
            context: 前 N 句英文原文（用于上下文消歧）

        Returns:
            中文译文
        """
        if self._model is None:
            raise RuntimeError("翻译模型未加载，请先调用 load_model()")

        # 空输入保护
        if not en_text or not en_text.strip():
            return ""

        # 拼接上下文
        if context and len(context) > 0:
            # 取最近 CONTEXT_SIZE 句
            recent = context[-CONTEXT_SIZE:]
            # opus-mt 用 " </s> " 作为上下文分隔符
            prompt = " </s> ".join(recent) + " </s> " + en_text
        else:
            prompt = en_text

        # 编码
        inputs = self._tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)

        # 动态 max_length：输入长度的 3 倍，最少 20，最多 200
        input_len = inputs.input_ids.shape[1]
        dyn_max_length = max(20, min(200, input_len * 3))

        # 生成
        outputs = self._model.generate(
            **inputs,
            max_length=dyn_max_length,
            num_beams=4,
            early_stopping=True,
            repetition_penalty=1.2,      # 抑制重复生成
            no_repeat_ngram_size=3,       # 禁止 3-gram 重复
        )

        # 解码
        result = self._tokenizer.decode(outputs[0], skip_special_tokens=True)

        # 清理
        result = result.strip()
        result = " ".join(result.split())

        # 后处理：折叠连续重复词（如 "你好 你好 你好" → "你好"）
        result = self._collapse_repetitions(result)

        return result

    def translate_batch(self, texts: List[str], context: Optional[List[str]] = None) -> List[str]:
        """批量翻译"""
        return [self.translate(t, context) for t in texts]

    @staticmethod
    def _collapse_repetitions(text: str) -> str:
        """
        折叠重复的中文词/短语/字符
        如 "你好 你好 你好" → "你好"
            "嗨 嗨嗨 嗨" → "嗨"
            "是 是" → "是"
        """
        if not text:
            return text

        words = text.split()
        if not words:
            return text

        # 词级去重：相邻相同词折叠
        collapsed = [words[0]]
        for w in words[1:]:
            if w != collapsed[-1]:
                # 也检查是否为子串重复（如 "嗨嗨" vs "嗨"）
                if w in collapsed[-1] or collapsed[-1] in w:
                    collapsed[-1] = max(w, collapsed[-1], key=len)
                else:
                    collapsed.append(w)

        # 字符级去重：忽略空格，仅比较 CJK 字符
        result = " ".join(collapsed)
        chars = list(result)
        deduped = []
        last_cjk = None  # 上一个 CJK 字符

        for c in chars:
            is_cjk = '一' <= c <= '鿿' or '぀' <= c <= 'ヿ' or '가' <= c <= '힯'
            if is_cjk:
                if c == last_cjk:
                    continue  # 跳过连续重复的 CJK 字符
                last_cjk = c
            else:
                last_cjk = None  # 非 CJK 字符重置
            deduped.append(c)

        return "".join(deduped)
