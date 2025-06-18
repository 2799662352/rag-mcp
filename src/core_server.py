from mcp.server.fastmcp import FastMCP
import torch
from FlagEmbedding import BGEM3FlagModel
import argparse
import logging
import time
import json
import numpy as np
from pathlib import Path
from typing import Optional, Dict, List, Any, Tuple, Union
import chromadb
from chromadb.config import Settings
import re
import traceback
from datetime import datetime
from typing import Dict, Any, List, Optional, Union
import sys

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 导入优化模块
try:
    from cache_system import global_cache
    from config_manager import config_manager
    from performance_monitor import global_monitor
    OPTIMIZATION_MODULES_LOADED = True
    logger.info("🚀 所有优化模块加载成功！")
except ImportError as e:
    logger.warning(f"⚠️ 优化模块导入失败: {e}")
    logger.warning("运行在兼容模式下，部分优化功能将不可用")
    OPTIMIZATION_MODULES_LOADED = False
    global_cache = None
    config_manager = None
    global_monitor = None

# 创建FastMCP实例
mcp = FastMCP("Danbooru搜索服务器-最小版-增强版")

# BGE-M3三重向量配置
CONFIG = {
    "max_length": 8192,
    "default_results": 20,
    "max_results": 200,
    "batch_size": 8,
    "database_mode": False,
    "nsfw_indicators": [
        "nude", "naked", "pussy", "sex", "cum", "nipples", "breast", "penis",
        "erection", "oral", "anal", "masturbation", "orgasm", "aroused", "horny",
        "裸体", "性", "阴", "阳具", "胸部", "乳头", "私处", "露出"
    ]
}

# 智能化系统变量
QUERY_HISTORY = []
QUERY_STATS = {}
USER_PREFERENCES = {}
PERFORMANCE_METRICS = {
    "total_queries": 0,
    "cache_hits": 0,
    "cache_misses": 0,
    "avg_response_time": 0.0,
    "last_query_time": None,
    "error_count": 0,
    "success_count": 0,
    "success_rate": 0.0
}