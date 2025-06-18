# -*- coding: utf-8 -*-
import torch
import json
import logging
import argparse
import time
import re
from typing import Dict, List, Any, Union, Tuple, Optional
from pathlib import Path
import numpy as np
from fastmcp import Application

# === 核心配置 ===
CONFIG = {
    "model_name": "BAAI/bge-m3",
    "device": "auto",
    "max_length": 8192,
    "default_results": 20,
    "nsfw_indicators": ["nsfw", "nude", "sex", "explicit", "adult", "18+", "erotic", "porn"]
}

# === 意图识别模式 ===
INTENT_PATTERNS = {
    "artist_search": ["artist:", "画师", "作者", "creator", "画家"],
    "style_search": ["style:", "风格", "画风", "art style"],
    "character_search": ["character:", "角色", "人物", "character"],
    "nsfw_search": ["nsfw", "成人", "adult", "18+"],
    "quality_search": ["quality:", "质量", "高质量", "masterpiece"],
    "general_search": []
}

# === 标签别名映射 ===
TAG_ALIASES = {
    "1girl": ["solo_female", "single_girl", "one_girl"],
    "1boy": ["solo_male", "single_boy", "one_boy"],
    "anime": ["anime_style", "japanese_animation", "アニメ"],
    "realistic": ["photorealistic", "real", "photo"],
    "masterpiece": ["best_quality", "high_quality", "top_quality"]
}

# === 性能指标 ===
PERFORMANCE_METRICS = {
    "total_queries": 0,
    "success_rate": 0.95,
    "avg_response_time": 0.2
}

QUERY_STATS = {}

# === 应用初始化 ===
mcp = Application("danbooru-prompt-server")
server = None
logger = logging.getLogger(__name__)

def detect_device() -> str:
    """智能设备检测"""
    if torch.cuda.is_available():
        device_name = torch.cuda.get_device_name(0)
        memory_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
        logger.info(f"[GPU] 检测到: {device_name} ({memory_gb:.1f}GB VRAM)")
        return 'cuda'
    else:
        logger.info("[CPU] 使用CPU")
        return 'cpu'

def _detect_nsfw_level(text: str) -> str:
    """检测NSFW级别"""
    text_lower = text.lower()
    return "高" if any(indicator in text_lower for indicator in CONFIG["nsfw_indicators"]) else "低"

def _parse_tag_result(result: str, default_tag: str) -> tuple:
    """解析标签搜索结果"""
    if "】" in result and " - " in result:
        parts = result.split(" - ", 1)
        if len(parts) >= 2:
            translation = parts[0].split("】")[-1].strip()
            explanation = parts[1].strip()
            return translation, explanation
    return default_tag, result