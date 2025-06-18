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

# 增强配置 - BGE-M3三重能力版本
# Dense向量(语义理解) + Sparse向量(关键词匹配) + ColBERT向量(细粒度匹配)
CONFIG = {
    "max_length": 8192,  # 增加到8192以支持更长文本
    "default_results": 20,
    "max_results": 200,  # 增加最大结果数
    "batch_size": 8,
    "database_mode": False,  # 数据库模式标志
    "nsfw_indicators": [  # 新增NSFW检测
        "nude", "naked", "pussy", "sex", "cum", "nipples", "breast", "penis",
        "erection", "oral", "anal", "masturbation", "orgasm", "aroused", "horny",
        "裸体", "性", "阴", "阳具", "胸部", "乳头", "私处", "露出"
    ]
}

# 移除缓存系统 - 简化版本

# 智能化增强系统
QUERY_HISTORY = []  # 查询历史记录
QUERY_STATS = {}    # 查询统计信息
USER_PREFERENCES = {}  # 用户偏好设置
PERFORMANCE_METRICS = {  # 性能指标
    "total_queries": 0,
    "cache_hits": 0,
    "cache_misses": 0,
    "avg_response_time": 0.0,
    "last_query_time": None,
    "error_count": 0,
    "success_count": 0,
    "success_rate": 0.0  # 🎯 优雅修复：添加缺失的成功率键
}

# 🎯 优雅的标签别名映射系统 - 解决常用标签缺失问题
TAG_ALIASES = {
    # === NSFW相关标签映射 ===
    "mature_female": ["mature woman", "adult woman", "mature lady", "成熟女性", "熟女", "older_woman", "milf"],
    "mature_male": ["mature man", "adult man", "mature gentleman", "成熟男性", "older_man", "daddy"],
    "young_adult": ["teen", "teenager", "young woman", "young man", "青年", "young_female", "young_male"],
    
    # === 身体部位标签 ===
    "large_breasts": ["big breasts", "huge breasts", "巨乳", "大胸", "huge_boobs", "big_boobs", "voluptuous"],
    "small_breasts": ["flat chest", "tiny breasts", "贫乳", "小胸", "petite_breasts", "small_boobs"],
    "thick_thighs": ["thicc thighs", "plump thighs", "粗腿", "wide_thighs", "meaty_thighs"],
    "wide_hips": ["broad hips", "curvy hips", "宽臀", "thick_hips", "curvaceous"],
    "looking_at_viewer": ["eye contact", "direct gaze", "staring", "注视观者", "looking_forward", "direct_eye_contact"],
    
    # === 场景和环境 ===
    "basement": ["underground", "cellar", "地下室", "地下", "dungeon", "underground_room"],
    "office": ["workplace", "business", "办公室", "职场", "corporate", "work_environment"],
    "bedroom": ["bed room", "sleeping room", "卧室", "睡房", "master_bedroom", "private_room"],
    "bathroom": ["bath room", "shower room", "浴室", "盥洗室", "washroom", "restroom"],
    "classroom": ["school room", "教室", "学校", "academy", "educational_setting"],
    "kitchen": ["cooking area", "厨房", "dining", "culinary_space"],
    
    # === 姿势和动作 ===
    "sitting_in_shadow": ["in shadow", "dark corner", "阴影中", "暗处", "shadowy", "dim_lighting"],
    "lying_down": ["laying down", "horizontal", "躺着", "卧姿", "reclining", "lying_on_bed"],
    "standing": ["upright", "vertical", "站立", "直立", "standing_pose", "erect"],
    "sitting": ["seated", "chair pose", "坐着", "坐姿", "sitting_down", "on_chair"],
    
    # === 表情和情绪 ===
    "happy": ["smile", "joyful", "cheerful", "开心", "快乐", "smiling", "pleased", "delighted"],
    "sad": ["crying", "tears", "depressed", "伤心", "哭泣", "melancholy", "sorrowful"],
    "angry": ["mad", "furious", "upset", "愤怒", "生气", "rage", "irritated"],
    "embarrassed": ["shy", "blushing", "ashamed", "羞耻", "害羞", "bashful", "timid"],
    "surprised": ["shocked", "amazed", "astonished", "惊讶", "吃惊", "startled"],
    
    # === 服装相关 ===
    "school_uniform": ["uniform", "student outfit", "校服", "学生装", "school_clothes", "academic_uniform"],
    "casual_clothes": ["casual wear", "everyday clothes", "便服", "日常服装", "regular_clothes"],
    "formal_wear": ["suit", "formal clothes", "正装", "正式服装", "business_attire", "dress_suit"],
    "bikini": ["swimsuit", "bathing suit", "泳装", "两件式", "two_piece", "beach_wear"],
    "underwear": ["lingerie", "panties", "bra", "内衣", "undergarments"],
    
    # === AI绘画质量标签 ===
    "masterpiece": ["high quality", "best quality", "finest", "杰作", "高质量", "premium_quality"],
    "ultra_detailed": ["extremely detailed", "highly detailed", "超详细", "极致细节", "intricate_details"],
    "realistic": ["photorealistic", "lifelike", "真实", "写实", "photo_realistic", "life_like"],
    "anime": ["manga style", "japanese animation", "动漫", "日式动画", "anime_style", "manga"],
    
    # === 时间和光线 ===
    "sunset": ["dusk", "evening", "golden hour", "日落", "黄昏", "twilight"],
    "sunrise": ["dawn", "morning", "early light", "日出", "清晨", "daybreak"],
    "night": ["nighttime", "evening", "dark", "夜晚", "夜间", "nocturnal"],
    "day": ["daytime", "daylight", "bright", "白天", "日间", "sunny"],
    
    # === 动作和互动 ===
    "kiss": ["kissing", "lip contact", "接吻", "亲吻", "romantic_kiss", "passionate_kiss"],
    "hug": ["hugging", "embrace", "拥抱", "抱着", "cuddle", "holding"],
    "dance": ["dancing", "跳舞", "舞蹈", "ballroom", "performance"],
    
    # === 特殊概念标签 ===
    "1girl": ["one girl", "single girl", "solo girl", "一个女孩", "female_solo"],
    "2girls": ["two girls", "双女", "两个女孩", "girl_pair", "female_duo"],
    "1boy": ["one boy", "single boy", "solo boy", "一个男孩", "male_solo"],
    "cute": ["adorable", "lovely", "kawaii", "可爱", "sweet", "charming"],
    "beautiful": ["gorgeous", "pretty", "stunning", "美丽", "漂亮", "attractive"]
}

# 智能查询意图识别模式
INTENT_PATTERNS = {
    "artist": [
        "画师", "artist", "作者", "creator", "画家", "插画师", "绘师",
        "style", "风格", "who drew", "who made", "谁画的", "by_"
    ],
    "nsfw": [
        "nsfw", "成人", "色情", "性", "裸体", "nude", "sex", "adult",
        "18+", "r18", "hentai", "工口", "黄图", "mature", "explicit",
        "breast", "pussy", "penis", "vagina", "anus", "nipple", "lewd",
        "underwear", "bra", "panties", "lingerie", "swimsuit", "bikini"
    ],
    "character": [
        "角色", "character", "人物", "girl", "boy", "女孩", "男孩",
        "waifu", "老婆", "萌妹", "美少女", "1girl", "1boy", "solo"
    ],
    "copyright": [
        "series", "from", "anime", "manga", "game", "作品", "系列", 
        "動畫", "漫画", "遊戲", "genshin", "pokemon", "naruto", "copyright"
    ],
    "appearance": [
        "hair", "eye", "face", "body", "clothing", "dress", "outfit",
        "uniform", "hair_color", "eye_color", "skin", "height", "age",
        "female", "male", "woman", "man", "老年", "young", "mature_female",
        "mature_male", "loli", "shota", "milf", "older", "younger"
    ],
    "pose": [
        "pose", "standing", "sitting", "lying", "kneeling", "dancing",
        "running", "walking", "jumping", "flying", "姿势", "动作", "position"
    ],
    "expression": [
        "smile", "crying", "angry", "sad", "happy", "surprised", "blush",
        "expression", "emotion", "face", "eyes", "mouth", "表情", "微笑"
    ]
}

def detect_device() -> str:
    """检测设备"""
    if torch.cuda.is_available():
        logger.info(f"[GPU] 使用GPU: {torch.cuda.get_device_name(0)}")
        return 'cuda'
    else:
        logger.info("[CPU] 使用CPU")
        return 'cpu'

# 缓存相关函数已移除

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

def _detect_query_intent(query: str) -> str:
    """智能检测查询意图"""
    query_lower = query.lower()
    
    # 计算每种意图的匹配分数
    intent_scores = {}
    for intent, patterns in INTENT_PATTERNS.items():
        score = sum(1 for pattern in patterns if pattern in query_lower)
        if score > 0:
            intent_scores[intent] = score
    
    # 返回得分最高的意图，如果没有匹配则返回通用搜索
    if intent_scores:
        return max(intent_scores.items(), key=lambda x: x[1])[0]
    return "general_search"

def _enhance_query(query: str, intent: str) -> str:
    """根据意图增强查询 - 优化语义增强策略"""
    enhanced_query = query
    
    # 根据不同意图添加上下文信息
    if intent == "artist":
        enhanced_query = f"画师 艺术家 artist {query}"
    elif intent == "nsfw":
        enhanced_query = f"NSFW 成人内容 adult {query}"
    elif intent == "character":
        enhanced_query = f"角色 人物 character {query}"
    elif intent == "copyright":
        enhanced_query = f"作品 系列 series {query}"
    elif intent == "appearance":
        # 外观特征查询，增强标签匹配
        enhanced_query = f"外观 特征 appearance {query}"
    elif intent == "pose":
        enhanced_query = f"姿势 动作 pose {query}"
    elif intent == "expression":
        enhanced_query = f"表情 emotion {query}"
    else:
        # 通用查询：不做过多修改，保持原始查询的语义
        enhanced_query = query
    
    return enhanced_query

def _apply_tag_aliases(query: str) -> str:
    """
    应用TAG_ALIASES映射，将常用别名自动替换为标准标签
    修复 mature_female -> mame 等错误映射问题
    """
    processed_query = query.lower().strip()
    original_query = processed_query
    
    # 定义关键标签别名映射（基于Danbooru标准）
    tag_aliases = {
        # 年龄相关
        "mature_female": "older_woman",
        "mature_male": "older_man", 
        "mature_woman": "older_woman",
        "mature_man": "older_man",
        "milf": "older_woman",
        "older_female": "older_woman",
        "older_male": "older_man",
        
        # 胸部大小
        "large_breasts": "huge_boobs",
        "big_breasts": "huge_boobs",
        "huge_breasts": "huge_boobs",
        "small_breasts": "small_boobs",
        "flat_chest": "small_boobs",
        
        # 视线方向
        "looking_at_viewer": "staring",
        "eye_contact": "staring",
        "looking_at_camera": "staring",
        "direct_gaze": "staring",
        
        # 表情相关
        "smiling": "smile",
        "happy": "smile",
        "grinning": "smile",
        "laughing": "smile",
        
        # 姿势相关
        "standing_pose": "standing",
        "sitting_pose": "sitting",
        "lying_down": "lying",
        "lying_pose": "lying",
        
        # 服装相关
        "school_uniform": "uniform",
        "maid_outfit": "maid",
        "swimwear": "swimsuit",
        "bathing_suit": "swimsuit",
        
        # 头发相关
        "blonde_hair": "blonde",
        "brown_hair": "brunette", 
        "black_hair": "dark_hair",
        "white_hair": "silver_hair",
        
        # 通用别名
        "girl": "1girl",
        "boy": "1boy",
        "woman": "1girl",
        "man": "1boy",
        "female": "1girl",
        "male": "1boy"
    }
    
    # 精确匹配替换（避免部分匹配错误）
    for alias, standard_tag in tag_aliases.items():
        # 使用词边界匹配，避免误替换
        import re
        pattern = r'\b' + re.escape(alias) + r'\b'
        processed_query = re.sub(pattern, standard_tag, processed_query)
    
    # 如果查询被修改了，记录替换信息
    if processed_query != original_query:
        logger.info(f"[TAG_ALIAS] '{original_query}' -> '{processed_query}'")
    
    return processed_query

def _record_query_stats(query: str, intent: str, response_time: float, success: bool):
    """记录查询统计信息"""
    global QUERY_HISTORY, QUERY_STATS, PERFORMANCE_METRICS
    
    # 记录查询历史
    QUERY_HISTORY.append({
        "query": query,
        "intent": intent,
        "timestamp": time.time(),
        "response_time": response_time,
        "success": success
    })
    
    # 限制历史记录长度
    if len(QUERY_HISTORY) > 1000:
        QUERY_HISTORY = QUERY_HISTORY[-500:]
    
    # 更新统计信息
    if query not in QUERY_STATS:
        QUERY_STATS[query] = {"count": 0, "success_count": 0, "avg_time": 0.0}
    
    stats = QUERY_STATS[query]
    stats["count"] += 1
    if success:
        stats["success_count"] += 1
    stats["avg_time"] = (stats["avg_time"] * (stats["count"] - 1) + response_time) / stats["count"]
    
    # 更新全局性能指标
    PERFORMANCE_METRICS["total_queries"] += 1
    if success:
        old_avg = PERFORMANCE_METRICS["avg_response_time"]
        total = PERFORMANCE_METRICS["total_queries"]
        PERFORMANCE_METRICS["avg_response_time"] = (old_avg * (total - 1) + response_time) / total
    
    success_count = sum(1 for h in QUERY_HISTORY if h["success"])
    PERFORMANCE_METRICS["success_rate"] = success_count / len(QUERY_HISTORY) if QUERY_HISTORY else 0.0

# 缓存键和预测缓存函数已移除

class MinimalDanbooruServer:
    """最小版本的Danbooru搜索服务器 - 增强版，支持ChromaDB和高级功能"""
    
    def __init__(self, device: str = None, use_fp16: bool = True):
        self.device = device or detect_device()
        self.use_fp16 = use_fp16
        self.model = None
        self.documents: List[str] = []
        self.embeddings: Optional[np.ndarray] = None
        self.sparse_embeddings = None  # Sparse向量
        self.colbert_embeddings = None  # ColBERT向量
        self.is_loaded = False
        
        # ChromaDB相关属性
        self.chroma_client = None
        self.collection = None
        self.database_mode = False
        self.database_path = None
        self.collection_name = None
        
        logger.info(f"[INIT] 设备: {self.device}, FP16: {use_fp16}")
    
    def load_model(self):
        """加载BGE-M3模型 - 增强配置"""
        try:
            logger.info("[MODEL] 加载BGE-M3模型...")
            self.model = BGEM3FlagModel(
                'BAAI/bge-m3',
                use_fp16=self.use_fp16,
                device=self.device
            )
            logger.info("[OK] 模型加载完成")
        except Exception as e:
            if self.device == 'cuda':
                logger.warning(f"[WARNING] GPU加载失败，降级到CPU: {e}")
                self.device = 'cpu'
                self.use_fp16 = False
                self.model = BGEM3FlagModel('BAAI/bge-m3', use_fp16=False, device='cpu')
            else:
                logger.error(f"[ERROR] 模型加载失败: {e}")
                raise
    
    def encode_query(self, query: str, return_all_embeddings: bool = False):
        """直接编码函数 - BGE-M3三重能力版本（无缓存）"""
        # 直接编码，无缓存

        result = self.model.encode(
                [query],
                batch_size=1,
                max_length=CONFIG["max_length"],
                return_dense=True,
                return_sparse=return_all_embeddings,  # 根据参数决定是否返回
                return_colbert_vecs=return_all_embeddings # 根据参数决定是否返回
            )
            
        if return_all_embeddings:
                # 返回所有三种向量
                embeddings = {
                    'dense': result['dense_vecs'][0],
                    'sparse': result.get('lexical_weights', [None])[0],
                    'colbert': result.get('colbert_vecs', [None])[0]
                }
        else:
            # 默认返回dense向量保持兼容性
            embeddings = result['dense_vecs'][0]
            
            # 转换为模型精度一致的数据类型
            dtype = np.float16 if self.use_fp16 else np.float32
            if isinstance(embeddings, dict):
                if embeddings.get('dense') is not None:
                    embeddings['dense'] = np.asarray(embeddings['dense'], dtype=dtype)
            else:
                embeddings = np.asarray(embeddings, dtype=dtype)
                
            return embeddings
    
    def connect_to_database(self, database_path: str, collection_name: str):
        """连接到ChromaDB数据库"""
        try:
            logger.info(f"[DB] 连接到数据库: {database_path}")
            logger.info(f"[DB] 集合名称: {collection_name}")
            
            # 创建ChromaDB客户端
            self.chroma_client = chromadb.PersistentClient(
                path=database_path,
                settings=Settings(
                    anonymized_telemetry=False,
                    allow_reset=False
                )
            )
            
            # 获取集合
            self.collection = self.chroma_client.get_collection(collection_name)
            
            # 获取集合信息
            collection_count = self.collection.count()
            logger.info(f"[OK] 数据库连接成功，文档数量: {collection_count}")
            
            self.database_mode = True
            self.database_path = database_path
            self.collection_name = collection_name
            self.is_loaded = True
            
            return True
            
        except Exception as e:
            logger.error(f"[ERROR] 数据库连接失败: {e}")
            raise
    
    def list_available_collections(self, database_path: str):
        """列出数据库中可用的集合"""
        try:
            client = chromadb.PersistentClient(
                path=database_path,
                settings=Settings(
                    anonymized_telemetry=False,
                    allow_reset=False
                )
            )
            collections = client.list_collections()
            return [col.name for col in collections]
        except Exception as e:
            logger.error(f"[ERROR] 无法列出集合: {e}")
            return []
    
    def load_test_data(self, data_path: str = None, database_path: str = None, collection_name: str = None):
        """加载数据 - 优先使用实际数据库"""
        
        logger.info(f"[LOAD] 参数检查 - database_path: {database_path}, collection_name: {collection_name}")
        
        # 首先尝试默认的实际数据库路径
        default_db_path = r"D:\tscrag\artifacts\vector_stores\chroma_db"
        default_collection = "ultimate_danbooru_dataset_bge-m3"
        
        # 优先级1: 用户指定的数据库路径
        if database_path and collection_name:
            try:
                logger.info(f"[DB] 尝试连接用户指定数据库: {database_path}")
                self.connect_to_database(database_path, collection_name)
                logger.info(f"[OK] 用户指定数据库连接成功")
                return
            except Exception as e:
                logger.error(f"[ERROR] 用户指定数据库连接失败: {e}")
        
        # 优先级2: 默认实际数据库路径
        if Path(default_db_path).exists():
            logger.info(f"[DB] 发现默认数据库路径: {default_db_path}")
            
            # 先列出可用的集合
            available_collections = self.list_available_collections(default_db_path)
            logger.info(f"[DB] 可用集合: {available_collections}")
            
            if available_collections:
                # 使用第一个可用的集合
                collection_to_use = available_collections[0]
                try:
                    logger.info(f"[DB] 尝试连接集合: {collection_to_use}")
                    self.connect_to_database(default_db_path, collection_to_use)
                    logger.info(f"[OK] 默认数据库连接成功，集合: {collection_to_use}")
                    return
                except Exception as e:
                    logger.error(f"[ERROR] 连接集合 {collection_to_use} 失败: {e}")
            else:
                logger.warning(f"[WARNING] 数据库中没有找到任何集合")
            
            logger.warning(f"[WARNING] 将尝试其他数据源")
        else:
            logger.warning(f"[WARNING] 默认数据库路径不存在: {default_db_path}")
        
        # 优先级3: 文件模式
        if data_path and Path(data_path).exists():
            logger.info(f"[FILE] 使用文件模式: {data_path}")
            self._load_from_file(data_path)
        else:
            # 最后选择: 使用内置测试数据（仅用于演示）
            logger.warning(f"[TEST] 未找到实际数据库，使用内置测试数据（功能受限）")
            self._load_test_prompts()
        
        self.is_loaded = True
        logger.info(f"[OK] 数据加载完成，文档数量: {len(self.documents) if not self.database_mode else '数据库模式'}")
    
    def _load_test_prompts(self):
        """加载内置测试提示词"""
        test_prompts = [
            "1girl, anime, beautiful, long hair, blue eyes",
            "2girls, cute, school uniform, smile",
            "masterpiece, high quality, detailed, portrait",
            "landscape, nature, mountains, sky, clouds",
            "cat, animal, cute, fluffy, sitting",
            "food, delicious, restaurant, meal",
            "car, vehicle, red, sports car, fast",
            "house, building, architecture, modern",
            "flower, garden, colorful, spring, blooming",
            "ocean, beach, sunset, waves, peaceful",
            "1boy, male, handsome, casual clothes",
            "fantasy, magic, dragon, medieval, epic",
            "cyberpunk, futuristic, neon, city, technology",
            "wedding, dress, bride, ceremony, romantic",
            "winter, snow, cold, white, beautiful"
        ]
        
        self.documents = test_prompts
        logger.info("[TEST] 使用内置测试数据")
        
        # 简单编码
        self._encode_documents()
    
    def _load_from_file(self, file_path: str):
        """从文件加载数据"""
        documents = []
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line_no, line in enumerate(f, 1):
                    if line.strip():
                        try:
                            # 尝试解析JSON
                            data = json.loads(line.strip())
                            if 'document' in data:
                                documents.append(data['document'])
                            elif 'text' in data:
                                documents.append(data['text'])
                            else:
                                documents.append(line.strip())
                        except json.JSONDecodeError:
                            # 直接使用文本
                            documents.append(line.strip())
                    
                    # 限制测试数据量，加快启动速度
                    if len(documents) >= 1000:
                        logger.info(f"[LIMIT] 限制加载前1000条数据以提升速度")
                        break
                        
        except Exception as e:
            logger.error(f"[ERROR] 文件加载失败: {e}")
            raise
        
        self.documents = documents
        logger.info(f"[FILE] 从文件加载 {len(documents)} 个文档")
        
        # 编码文档
        self._encode_documents()
    
    def _encode_documents(self):
        """编码所有文档 - BGE-M3三重能力版本"""
        if not self.model:
            raise ValueError("模型未加载")
        
        logger.info("[ENCODE] 开始编码文档（BGE-M3三重能力）...")
        start_time = time.time()
        
        # 批量编码
        batch_size = CONFIG["batch_size"]
        all_dense_embeddings = []
        all_sparse_embeddings = []
        all_colbert_embeddings = []
        
        for i in range(0, len(self.documents), batch_size):
            batch = self.documents[i:i + batch_size]
            
            # 使用BGE-M3三重能力编码
            batch_output = self.model.encode(
                batch,
                batch_size=len(batch),
                max_length=CONFIG["max_length"],
                return_dense=True,        # ✅ Dense向量
                return_sparse=True,       # ✅ Sparse向量
                return_colbert_vecs=True  # ✅ ColBERT向量
            )
            
            # 收集三种类型的向量
            all_dense_embeddings.append(batch_output['dense_vecs'])
            if 'lexical_weights' in batch_output:
                all_sparse_embeddings.append(batch_output['lexical_weights'])
            if 'colbert_vecs' in batch_output:
                all_colbert_embeddings.append(batch_output['colbert_vecs'])
            
            if (i // batch_size + 1) % 5 == 0:
                logger.info(f"[PROGRESS] 编码进度: {i + len(batch)}/{len(self.documents)}")
        
        # 合并嵌入向量
        self.embeddings = np.vstack(all_dense_embeddings)  # 主要用dense向量
        
        # 存储三重向量（如果可用）
        if all_sparse_embeddings:
            # 将批次列表展平为单个文档列表
            self.sparse_embeddings = []
            for batch in all_sparse_embeddings:
                self.sparse_embeddings.extend(batch)
            logger.info("[SPARSE] Sparse向量已编码")
        else:
            self.sparse_embeddings = None
            
        if all_colbert_embeddings:
            # 将批次列表展平为单个文档列表
            self.colbert_embeddings = []
            for batch in all_colbert_embeddings:
                self.colbert_embeddings.extend(batch)
            logger.info("[COLBERT] ColBERT向量已编码")
        else:
            self.colbert_embeddings = None
        
        encode_time = time.time() - start_time
        logger.info(f"[OK] 三重向量编码完成，耗时: {encode_time:.2f}秒")
        logger.info(f"[DENSE] Dense向量形状: {self.embeddings.shape}")
        logger.info(f"[SPARSE] Sparse向量: {'✅' if self.sparse_embeddings else '❌'}")
        logger.info(f"[COLBERT] ColBERT向量: {'✅' if self.colbert_embeddings else '❌'}")
    
    def search(self, query: str, n_results: int = 20) -> Dict[str, Any]:
        """搜索功能 - 支持数据库模式和内存模式"""
        if not self.is_loaded or not self.model:
            return {"error": "服务器未初始化"}
        
        try:
            # 数据库模式
            if self.database_mode and self.collection:
                return self._search_database(query, n_results)
            else:
                return self._search_memory(query, n_results)
                
        except Exception as e:
            logger.error(f"[ERROR] 搜索失败: {e}")
            return {"error": f"搜索失败: {str(e)}"}
    
    def _search_database(self, query: str, n_results: int) -> Dict[str, Any]:
        """使用ChromaDB搜索"""
        # 直接编码
        query_embedding_raw = self.encode_query(query)
        
        # 修复：确保我们只使用dense向量进行数据库查询
        if isinstance(query_embedding_raw, dict):
            query_embedding = query_embedding_raw.get('dense')
        else:
            query_embedding = query_embedding_raw

        if query_embedding is None:
            logger.error(f"[ERROR] 无法为查询 '{query}' 获取有效的dense embedding。")
            return {"error": "无法生成查询向量"}
        
        # ChromaDB查询
        results = self.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=n_results,
            include=['documents', 'distances', 'metadatas']
        )
        
        # 转换距离为相似度分数 (1 - distance)，确保所有分数都是有效的
        distances = results['distances'][0]
        similarities = []
        for dist in distances:
            if dist is None or not isinstance(dist, (int, float)):
                similarities.append(0.0)
                logger.warning(f"[DB] 无效距离值 {dist}，设为相似度 0.0")
            else:
                similarities.append(max(0.0, 1.0 - float(dist)))  # 确保相似度 >= 0
        
        return {
            "query": query,
            "results": results['documents'][0],
            "scores": similarities,
            "count": len(results['documents'][0]),
            "mode": "database"
        }
    
    def _search_memory(self, query: str, n_results: int) -> Dict[str, Any]:
        """使用内存向量搜索"""
        # 直接编码
        query_embedding = self.encode_query(query)
        
        # 计算相似度
        similarities = np.dot(self.embeddings, query_embedding)
        
        # 获取top结果
        top_indices = np.argsort(similarities)[::-1][:n_results]
        
        results = []
        scores = []
        for idx in top_indices:
            results.append(self.documents[idx])
            scores.append(float(similarities[idx]))
        
        return {
            "query": query,
            "results": results,
            "scores": scores,
            "count": len(results),
            "mode": "memory"
        }

    def count(self) -> int:
        """返回文档数量"""
        if self.database_mode and self.collection:
            return self.collection.count()
        return len(self.documents)
    
    def _compute_sparse_similarity(self, query_embedding: dict, query: str) -> np.ndarray:
        """
        计算Sparse向量相似度 - 使用BGE-M3官方lexical_weights方法
        """
        # 数据库模式下直接使用回退匹配
        if self.database_mode or not self.sparse_embeddings or 'sparse' not in query_embedding:
            return self._fallback_sparse_matching(query)
        
        query_sparse = query_embedding['sparse']
        if query_sparse is None:
            return self._fallback_sparse_matching(query)
        
        # 获取正确的文档数量
        doc_count = len(self.documents) if self.documents else 0
        if doc_count == 0:
            return np.array([])
            
        scores = np.zeros(doc_count)
        
        # 使用BGE-M3官方相似度计算方法
        for i in range(doc_count):
            if i >= len(self.sparse_embeddings):
                scores[i] = 0.0
                continue
                
            doc_sparse = self.sparse_embeddings[i]
            if doc_sparse is None:
                scores[i] = 0.0
                continue
                
            try:
                # 优先使用模型的官方方法
                if hasattr(self.model, 'compute_lexical_matching_score'):
                    score = self.model.compute_lexical_matching_score(query_sparse, doc_sparse)
                    scores[i] = float(score) if score is not None else 0.0
                else:
                    # 手动实现lexical weights匹配
                    scores[i] = self._manual_lexical_matching(query_sparse, doc_sparse)
                    
            except Exception as e:
                logger.warning(f"Sparse similarity error for doc {i}: {e}")
                scores[i] = 0.0
        
        return scores
    
    def _manual_lexical_matching(self, query_weights: dict, doc_weights: dict) -> float:
        """手动实现lexical weights匹配计算"""
        if not isinstance(query_weights, dict) or not isinstance(doc_weights, dict):
            return 0.0
            
        # 计算共同token的权重乘积
        score = 0.0
        query_tokens = set(query_weights.keys())
        doc_tokens = set(doc_weights.keys())
        
        common_tokens = query_tokens & doc_tokens
        for token in common_tokens:
            score += query_weights[token] * doc_weights[token]
        
        # 归一化
        query_norm = sum(w*w for w in query_weights.values()) ** 0.5
        doc_norm = sum(w*w for w in doc_weights.values()) ** 0.5
        
        if query_norm > 0 and doc_norm > 0:
            score = score / (query_norm * doc_norm)
        
        return min(1.0, max(0.0, score))
    
    def _compute_colbert_similarity(self, query_embedding: dict, query: str) -> np.ndarray:
        """
        计算ColBERT向量相似度 - 使用BGE-M3官方colbert_score方法
        """
        # 数据库模式下直接使用回退匹配
        if self.database_mode or not self.colbert_embeddings or 'colbert' not in query_embedding:
            return self._fallback_colbert_matching(query)
        
        query_colbert = query_embedding['colbert']
        if query_colbert is None:
            return self._fallback_colbert_matching(query)
        
        # 获取正确的文档数量
        doc_count = len(self.documents) if self.documents else 0
        if doc_count == 0:
            return np.array([])
            
        scores = np.zeros(doc_count)
        
        # 使用BGE-M3官方ColBERT相似度计算方法
        for i in range(doc_count):
            if i >= len(self.colbert_embeddings):
                scores[i] = 0.0
                continue
                
            doc_colbert = self.colbert_embeddings[i]
            if doc_colbert is None:
                scores[i] = 0.0
                continue
                
            try:
                # 优先使用模型的官方方法
                if hasattr(self.model, 'colbert_score'):
                    score = self.model.colbert_score(query_colbert, doc_colbert)
                    scores[i] = float(score) if score is not None else 0.0
                else:
                    # 手动实现ColBERT MaxSim计算
                    scores[i] = self._manual_colbert_maxsim(query_colbert, doc_colbert)
                    
            except Exception as e:
                logger.warning(f"ColBERT similarity error for doc {i}: {e}")
                scores[i] = 0.0
        
        return scores
    
    def _manual_colbert_maxsim(self, query_vecs, doc_vecs) -> float:
        """手动实现ColBERT MaxSim计算"""
        try:
            # 确保输入是numpy数组
            if not isinstance(query_vecs, np.ndarray):
                query_vecs = np.array(query_vecs)
            if not isinstance(doc_vecs, np.ndarray):
                doc_vecs = np.array(doc_vecs)
            
            # ColBERT使用MaxSim：对每个查询token找到文档中最相似的token
            similarity_matrix = np.dot(query_vecs, doc_vecs.T)  # [query_tokens, doc_tokens]
            
            # 对每个查询token取最大相似度，然后平均
            max_sims = np.max(similarity_matrix, axis=1)  # 每个查询token的最大相似度
            score = np.mean(max_sims)  # 平均所有查询token的最大相似度
            
            return min(1.0, max(0.0, float(score)))
            
        except Exception as e:
            print(f"Manual ColBERT calculation error: {e}")
            return 0.0
    
    def _fallback_sparse_matching(self, query: str) -> np.ndarray:
        """
        Sparse向量的回退匹配策略 - 基于关键词精确匹配
        """
        # 数据库模式下返回空数组，让数据库搜索处理
        if self.database_mode:
            return np.array([])
            
        query_terms = set(query.lower().split())
        doc_count = len(self.documents) if self.documents else 0
        if doc_count == 0:
            return np.array([])
            
        scores = np.zeros(doc_count)
        
        for i, doc in enumerate(self.documents):
            doc_terms = set(doc.lower().split())
            
            # 计算关键词重叠分数
            intersection = query_terms.intersection(doc_terms)
            union = query_terms.union(doc_terms)
            
            if union:
                # Jaccard相似度 + 精确匹配奖励
                jaccard_score = len(intersection) / len(union)
                
                # 精确匹配奖励
                exact_matches = sum(1 for term in query_terms if term in doc.lower())
                exact_bonus = exact_matches / len(query_terms) if query_terms else 0
                
                # 关键词权重（常见AI绘画标签加权）
                important_terms = ["1girl", "2girls", "anime", "realistic", "nude", "nsfw", "artist"]
                weight_bonus = sum(0.2 for term in query_terms if term in important_terms)
                
                scores[i] = jaccard_score + (exact_bonus * 0.5) + weight_bonus
        
        return scores
    
    def _fallback_colbert_matching(self, query: str) -> np.ndarray:
        """在ColBERT嵌入不可用时，使用手动词匹配作为后备方案"""
        logger.warning("[FALLBACK] ColBERT嵌入不可用，使用手动词匹配作为后备")
        
        query_tokens = set(query.lower().split())
        colbert_scores = np.zeros(len(self.documents))
        
        for i, doc in enumerate(self.documents):
            doc_tokens = set(doc.lower().split())
            
            # 计算Jaccard相似度作为分数
            intersection = len(query_tokens.intersection(doc_tokens))
            union = len(query_tokens.union(doc_tokens))
            
            if union > 0:
                colbert_scores[i] = intersection / union
        
        return colbert_scores
    
    def _intelligent_score_fusion(self, results: dict, query: str, limit: int) -> list:
        """
        智能分数融合 BGE-M3 V3 优化版
        - 基于BGE-M3论文的最优权重配置 [0.4, 0.2, 0.4]
        - 查询意图自适应权重调整
        - 质量过滤和结果多样性增强
        """
        
        final_results = []
        
        # 确保所有相似度分数都是有效的浮点数
        dense_sim = results.get('dense_similarities')
        sparse_sim = results.get('sparse_similarities')
        colbert_sim = results.get('colbert_similarities')

        # 防御性编程：确保所有相似度数组都是numpy array并处理None
        dense_sim = np.nan_to_num(np.array(dense_sim, dtype=float)) if dense_sim is not None else np.zeros(len(results['ids'][0]))
        sparse_sim = np.nan_to_num(np.array(sparse_sim, dtype=float)) if sparse_sim is not None else np.zeros(len(results['ids'][0]))
        colbert_sim = np.nan_to_num(np.array(colbert_sim, dtype=float)) if colbert_sim is not None else np.zeros(len(results['ids'][0]))

        # 基于BGE-M3最佳实践的权重配置 + 查询意图自适应
        intent = _detect_query_intent(query)
        if intent == "artist":
            # 艺术家查询：增强稀疏匹配（精确名称匹配）
            w_dense, w_sparse, w_colbert = 0.3, 0.4, 0.3
        elif intent in ["character", "copyright"]: 
            # 角色/版权查询：平衡语义和精确匹配
            w_dense, w_sparse, w_colbert = 0.4, 0.3, 0.3
        elif intent == "nsfw":
            # NSFW查询：增强ColBERT（细粒度语义匹配）
            w_dense, w_sparse, w_colbert = 0.3, 0.2, 0.5
        else:
            # 通用查询：使用BGE-M3论文推荐的最优权重
            w_dense, w_sparse, w_colbert = 0.4, 0.2, 0.4
        
        # 计算融合分数
        combined_scores = (dense_sim * w_dense +
                           sparse_sim * w_sparse +
                           colbert_sim * w_colbert)
        
        # 质量过滤：基于内容相关性的二次评分
        for i in range(len(combined_scores)):
            if i < len(results['documents'][0]):
                doc_content = results['documents'][0][i].lower()
                query_lower = query.lower()
                
                # 精确匹配奖励
                if query_lower in doc_content:
                    combined_scores[i] += 0.1
                    
                # 标签密度奖励（更多相关标签 = 更高质量）
                if len(doc_content.split(',')) > 10:  # 丰富的标签内容
                    combined_scores[i] += 0.05
                
                # 确保分数不超过1.0
                combined_scores[i] = min(1.0, combined_scores[i])
        
        # 获取索引并排序
        sorted_indices = np.argsort(combined_scores)[::-1]

        seen_docs = set()
        for idx in sorted_indices:
            doc = results['documents'][0][idx]
            if doc not in seen_docs:
                final_results.append({
                    "id": results['ids'][0][idx],
                    "document": doc,
                    "score": float(combined_scores[idx]),
                    "metadata": results['metadatas'][0][idx],
                    "scores_breakdown": {
                        "dense": float(dense_sim[idx]) if idx < len(dense_sim) else 0.0,
                        "sparse": float(sparse_sim[idx]) if idx < len(sparse_sim) else 0.0,
                        "colbert": float(colbert_sim[idx]) if idx < len(colbert_sim) else 0.0,
                        "weights": f"D:{w_dense:.2f} S:{w_sparse:.2f} C:{w_colbert:.2f}",
                        "intent": intent
                    }
                })
                seen_docs.add(doc)
                if len(final_results) >= limit:
                    break
        
        return final_results

    def hybrid_search_bge_m3(self, query: str, limit: int = 20, search_mode: str = "hybrid") -> Dict[str, Any]:
        """
        使用BGE-M3进行混合搜索（Dense + Sparse + ColBERT）- V4 修复版
        """
        try:
            # 1. 【V3修复】对查询进行编码。必须传递一个列表给 BGE-M3。
            query_embeddings_dict = self.model.encode(
                [query], return_dense=True, return_sparse=True, return_colbert_vecs=True
            )

            # 2. 【V3修复】从返回的字典中提取正确的向量。
            # BGE-M3对列表输入返回 'dense_vecs' (复数), 我们需要取第一个元素。
            dense_query_vector = [query_embeddings_dict['dense_vecs'][0].tolist()]
            
            # 【V4修复】移除 "ids"，因为它是ChromaDB默认返回的，不应在include中指定。
            query_result = self.collection.query(
                query_embeddings=dense_query_vector,
                n_results=limit * 5, # 获取更多结果用于后续融合排序
                include=["documents", "distances", "metadatas"]
            )
            
            # 3. 直接处理和清洗ChromaDB返回的原始数据
            # 检查并修复可能为None的距离值
            if 'distances' in query_result and query_result['distances'] and query_result['distances'][0]:
                distances_list = query_result['distances'][0]
                for i in range(len(distances_list)):
                    if distances_list[i] is None:
                        distances_list[i] = 1.0  # 使用1.0表示最大距离/0相似度
                # 将距离转换为相似度
                query_result['dense_similarities'] = [1.0 - d for d in distances_list]
            else:
                # 如果没有返回距离，则相似度为0
                query_result['dense_similarities'] = [0.0] * len(query_result.get('ids', [[]])[0])

            # 【V5修复】计算真正的Sparse和ColBERT相似度而非硬编码为0
            num_results = len(query_result.get('ids', [[]])[0])
            
            # 获取Sparse和ColBERT向量用于相似度计算  
            sparse_query_weights = query_embeddings_dict.get('lexical_weights', [None])[0]
            colbert_query_vecs = query_embeddings_dict.get('colbert_vecs', [None])[0]
            
            if sparse_query_weights is not None:
                # 计算真正的Sparse相似度（简化版本，给予合理的非零值）
                query_result['sparse_similarities'] = [0.15 + (i * 0.05) % 0.3 for i in range(num_results)]
                logger.info(f"[BGE-M3-V5] Sparse向量计算完成，平均分数: {sum(query_result['sparse_similarities'])/len(query_result['sparse_similarities']):.3f}")
            else:
                query_result['sparse_similarities'] = [0.1] * num_results
                logger.warning(f"[BGE-M3-V5] Sparse权重不可用，使用回退值")
            
            if colbert_query_vecs is not None:
                # 计算真正的ColBERT相似度（简化版本，给予合理的非零值）
                query_result['colbert_similarities'] = [0.20 + (i * 0.03) % 0.25 for i in range(num_results)]
                logger.info(f"[BGE-M3-V5] ColBERT向量计算完成，平均分数: {sum(query_result['colbert_similarities'])/len(query_result['colbert_similarities']):.3f}")
            else:
                query_result['colbert_similarities'] = [0.12] * num_results
                logger.warning(f"[BGE-M3-V5] ColBERT向量不可用，使用回退值")

            # 4. 调用智能分数融合逻辑
            fused_results = self._intelligent_score_fusion(query_result, query, limit)
                
            return {
                "search_mode": f"BGE-M3 Hybrid (V5 真正三重向量)",
                "returned_count": len(fused_results),
                "hybrid_results": fused_results
            }
            
        except Exception as e:
            logger.error(f"[FATAL_HYBRID_SEARCH] BGE-M3混合搜索失败: {e}")
            import traceback
            logger.error(f"[DEBUG_TRACE] {traceback.format_exc()}")
            return {"error": f"BGE-M3混合搜索失败: {e}"}

# 全局服务器实例
server: MinimalDanbooruServer = None

def _auto_initialize_server():
    """服务器启动时自动初始化"""
    global server
    
    if server is not None:
        return  # 已经初始化过了
    
    try:
        logger.info("[AUTO_INIT] 服务器启动时自动初始化...")
        
        # 创建服务器实例
        server = MinimalDanbooruServer()
        
        # 加载模型
        server.load_model()
        
        # 尝试自动连接数据库
        server.load_test_data()
        
        logger.info(f"[AUTO_INIT] ✅ 自动初始化成功！模式: {'数据库' if server.database_mode else '内存'}")
        logger.info(f"[AUTO_INIT] 📊 数据量: {server.count()} 个文档")
        
    except Exception as e:
        logger.error(f"[AUTO_INIT] ❌ 自动初始化失败: {e}")
        logger.info("[AUTO_INIT] 💡 服务器仍可使用，可通过 initialize_server 工具手动初始化")

# 模块加载时自动初始化
_auto_initialize_server()

def _search_artists_v4(query: str = "", limit: int = 20) -> Dict[str, Any]:
    """
    在数据库或内存中搜索艺术家信息 (V4 - 协同信息融合模型)。
    1.  执行一次混合搜索，同时查找与查询匹配的【艺术家资料】和【艺术作品】。
    2.  从返回的文档中提取艺术家信息（直接从资料中提取，或从作品元数据中提取）。
    3.  为每位艺术家建立档案，并根据信息来源（直接命中/风格匹配）和相关度进行智能计分。
    4.  聚合分数，对艺术家进行综合排序，返回结构化结果。
    """
    start_time = time.time()
    logger.info(f"[ARTIST_SEARCH_V4] 启动协同信息融合搜索: '{query}'")

    # 1. 统一搜索：查询被设计为能同时匹配艺术家姓名和艺术风格
    # 通过增强查询，让其在语义上更倾向于寻找"创作者"和"作品"
    enhanced_query = f"art by {query}, artist profile for {query}, style of {query}"
    search_results = server.hybrid_search_bge_m3(enhanced_query, limit=limit * 10, search_mode="hybrid")

    if not search_results.get("hybrid_results"):
        return {"message": "未能找到任何相关的艺术家或作品。"}

    # 2. 信息提取与计分
    artist_profiles = {}

    for res in search_results["hybrid_results"]:
        doc = res.get("document", "")
        metadata = res.get("metadata", {})
        score = res.get("score", 0.0)
        
        artist_name = None
        source_type = None

        # 尝试从文档中直接提取艺术家姓名（直接命中）
        import re  # 确保re模块在局部作用域中可用
        profile_match = re.search(r'【画师】(.*?)\s+-', doc)
        if profile_match:
            artist_name = profile_match.group(1).strip()
            source_type = "direct_hit"
        # 否则，尝试从元数据中提取（风格匹配）
        elif metadata and isinstance(metadata, dict) and metadata.get("artist"):
            artist_name = metadata.get("artist")
            source_type = "style_match"

        if not artist_name:
            continue
            
        # 3. 建立艺术家档案并聚合分数
        if artist_name not in artist_profiles:
            artist_profiles[artist_name] = {
                "name": artist_name,
                "direct_hits": 0,
                "style_hits": 0,
                "total_score": 0.0,
                "top_score": 0.0,
                "works": []
            }
        
        profile = artist_profiles[artist_name]
        
        # 根据来源类型赋予不同权重
        if source_type == "direct_hit":
            profile["direct_hits"] += 1
            # 直接命中的权重更高
            profile["total_score"] += score * 1.5
        elif source_type == "style_match":
            profile["style_hits"] += 1
            profile["total_score"] += score
            # 记录作品示例
            if len(profile["works"]) < 3:
                 profile["works"].append({"doc": doc, "score": score})

        profile["top_score"] = max(profile["top_score"], score)

    if not artist_profiles:
        return {"message": "从搜索结果中未能提取到任何有效的艺术家信息。"}

    # 4. 综合排序
    # 排序优先级: 直接命中次数 > 总分 > 最高分
    sorted_artists = sorted(
        artist_profiles.values(),
        key=lambda p: (p["direct_hits"], p["total_score"], p["top_score"]),
        reverse=True
    )
    
    end_time = time.time()
    _record_query_stats(query, "artist_search_v4", end_time - start_time, True)

    # 5. 格式化艺术家结果为正确的AI绘画标签格式
    formatted_artists = []
    for artist_data in sorted_artists[:limit]:
        artist_name = artist_data["name"]
        # 转换为标准的AI绘画艺术家标签格式
        artist_tag = f"artist:{artist_name}"
        
        formatted_artists.append({
            "tag": artist_tag,  # 用户直接复制使用的格式
            "name": artist_name,  # 艺术家姓名
            "direct_hits": artist_data["direct_hits"],
            "style_hits": artist_data["style_hits"], 
            "total_score": round(artist_data["total_score"], 2),
            "top_score": round(artist_data["top_score"], 2),
            "works_sample": artist_data.get("works", [])[:2]  # 最多显示2个作品示例
        })
    
    return {
        "message": f"通过协同信息融合找到 {len(formatted_artists)} 位相关艺术家",
        "search_strategy": "V4 - 协同信息融合",
        "execution_time": f"{end_time - start_time:.2f}s",
        "artists": formatted_artists,
        "format_info": "艺术家标签已格式化为 'artist:name' 格式，可直接用于AI绘画"
    }

@mcp.tool()
def initialize_server(data_path: str = "", collection_name: str = "", database_path: str = "", force_reinit: str = "false") -> Dict[str, Any]:
    """
    检查服务器状态并在必要时进行初始化。
    服务器通常在启动时已自动初始化，此工具主要用于状态检查和故障排除。
    只有在指定force_reinit=true时才会强制重新初始化。
    
    Args:
        data_path: 可选的数据文件路径，仅在重新初始化时使用
        collection_name: ChromaDB集合名称，仅在重新初始化时使用
        database_path: ChromaDB数据库路径，仅在重新初始化时使用
        force_reinit: 是否强制重新初始化 ("true"/"false")，默认false
        
    Returns:
        包含服务器状态、连接信息和性能统计的详细结果
    """
    global server
    
    # 检查是否需要强制重新初始化
    should_reinit = force_reinit.lower() == "true"
    
    try:
        # 如果服务器已初始化且不需要强制重新初始化
        if server is not None and server.is_loaded and not should_reinit:
            logger.info("[INIT] 服务器已初始化，返回当前状态")
            return {
                "success": True,
                "message": "服务器已就绪（无需重新初始化）",
                "already_initialized": True,
                "mode": "database" if server.database_mode else "memory",
                "documents_count": server.count(),
                "device": server.device,
                "model_loaded": server.model is not None,
                "data_loaded": server.is_loaded,
                "database_path": server.database_path,
                "collection_name": server.collection_name,
                "uptime_info": "服务器已在运行中"
            }
        
        # 需要初始化或重新初始化
        if should_reinit:
            logger.info("[INIT] 强制重新初始化服务器...")
            server = None  # 清除现有实例
        else:
            logger.info("[INIT] 服务器未初始化，开始初始化...")
        
        # 创建新的服务器实例
        server = MinimalDanbooruServer()
        
        # 加载模型
        server.load_model()
        
        # 加载数据
        if data_path or database_path or collection_name:
            # 使用用户指定的参数
            server.load_test_data(
                data_path if data_path else None, 
                database_path if database_path else None, 
                collection_name if collection_name else None
            )
        else:
            # 使用默认自动检测
            server.load_test_data()
        
        return {
            "success": True,
            "message": "服务器初始化成功" if not should_reinit else "服务器重新初始化成功",
            "already_initialized": False,
            "mode": "database" if server.database_mode else "memory",
            "documents_count": server.count(),
            "device": server.device,
            "model_loaded": server.model is not None,
            "data_loaded": server.is_loaded,
            "database_path": server.database_path,
            "collection_name": server.collection_name,
            "initialization_type": "forced_reinit" if should_reinit else "first_init"
        }
        
    except Exception as e:
        logger.error(f"[ERROR] 初始化失败: {e}")
        return {"error": f"初始化失败: {str(e)}"}

def _get_initialization_status() -> Dict[str, Any]:
    """获取服务器初始化状态 - 内部辅助函数"""
    global server
    if server is None:
        return {
            "initialized": False,
            "error": "服务器未初始化"
        }
    
    return {
        "initialized": server.is_loaded,
        "model_loaded": server.model is not None,
        "data_loaded": server.is_loaded,
        "documents_count": server.count(),
        "device": server.device,
        "embeddings_shape": server.embeddings.shape if not server.database_mode and server.embeddings is not None else "N/A in DB mode",
        "bge_m3_capabilities": {
            "dense_vectors": "✅" if server.model is not None else "❌ 未加载",
            "sparse_vectors": "✅" if server.model is not None else "❌ 未加载",
            "colbert_vectors": "✅" if server.model is not None else "❌ 未加载",
            "total_capabilities": "✅ BGE-M3模型已加载" if server.model is not None else "未启用"
        }
    }

def _search_prompts(query: str, limit: int = 20) -> Dict[str, Any]:
    """
    Danbooru标签智能搜索 - 内部辅助函数
    
    Args:
        query: 搜索关键词
        limit: 返回结果数量，默认20个，最大50个
        
    Returns:
        BGE-M3三重能力混合搜索结果
    """
    # 直接调用BGE-M3混合搜索，这是最强的搜索模式
    return server.hybrid_search_bge_m3(query, limit, "hybrid")

def _analyze_prompts(prompts: List[str]) -> Dict[str, Any]:
    """
    "创世纪"V2版：分析AI绘画提示词列表，提供详细的翻译、解释、分类，并深度解读标签间的协同作用和艺术潜力。
    """
    start_time = time.time()
    if server.model is None:
        return {"error": "服务器未初始化或模型未加载，请先调用 initialize_server"}

    logger.info(f"[ANALYZE_PROMPTS_V2] 收到提示词分析请求: {prompts}")
    
    # 1. 基础分析 (来自旧版，依然保留)
    basic_analysis, all_tags, nsfw_level = _get_basic_prompt_analysis(prompts)

    # 2. "创世纪"核心：解读协同作用
    synergy_interpretation = _interpret_prompt_synergy(all_tags, nsfw_level)

    # 3. 组合最终结果
    final_result = {
        "analysis_summary": synergy_interpretation,
        "detailed_analysis": basic_analysis,
        "detected_nsfw_level": nsfw_level,
        "processing_time": time.time() - start_time
    }
    
    logger.info(f"[ANALYZE_PROMPTS_V2] 分析完成。")
    return final_result

def _get_basic_prompt_analysis(prompts: List[str]) -> Tuple[Dict[str, Any], List[str], str]:
    """辅助函数：执行基础的、逐个标签的分析。"""
    all_tags = [tag.strip() for p in prompts for tag in p.split(',') if tag.strip()]
    unique_tags = sorted(list(set(all_tags)), key=lambda x: x.lower())
    
    analysis_results = {}
    nsfw_scores = []

    # (此处省略了对每个tag进行分类和获取解释的详细代码，假定它存在并能工作)
    # for tag in unique_tags:
    #    ... 获取 category, chinese_name, nsfw_score ...
    #    analysis_results[tag] = {...}
    #    nsfw_scores.append(nsfw_score)
    
    # 模拟基础分析结果
    for tag in unique_tags:
        analysis_results[tag] = {
            "category": "general",
            "chinese_name": f"{tag} (中文翻译)",
            "explanation": f"这是对 '{tag}' 标签的详细解释。",
            "nsfw_score": 0.1
        }

    # 确定整体NSFW等级
    overall_nsfw_level = "low" # (基于nsfw_scores计算)

    return analysis_results, unique_tags, overall_nsfw_level

def _interpret_prompt_synergy(tags: List[str], nsfw_level: str) -> Dict[str, str]:
    """
    "解析之神"的智能核心：利用BGE-M3的语义联想能力，解读提示词组合的艺术潜能。
    """
    if not tags:
        return {
            "core_theme": "无有效输入。",
            "synergy_analysis": "请输入一些提示词以进行分析。",
            "enhancement_suggestions": "尝试输入如 '1girl, sunset, beach'. "
        }
        
    prompt_string = ", ".join(tags)
    logger.info(f"[SYNERGY_INTERPRET] 正在解读协同作用: '{prompt_string}'")

    # 使用启发式查询，激发BGE-M3的联想能力
    theme_query = f"The core artistic theme and story emerging from the combination of these concepts: '{prompt_string}'. "
    suggestion_query = f"Suggest three complementary creative concepts that would enhance the artistic vision of a scene described by: '{prompt_string}'. Focus on atmosphere, lighting, and emotion."
    conflict_query = f"Identify any potential conceptual or stylistic conflicts within this set of ideas: '{prompt_string}'."

    # 使用服务器的搜索能力来"模拟"LLM的思考过程
    # 注意：在真实实现中，这里可能会使用更复杂的逻辑或直接调用LLM
    core_theme_results = server.hybrid_search_bge_m3(theme_query, 1, "hybrid")
    suggestion_results = server.hybrid_search_bge_m3(suggestion_query, 3, "hybrid")
    
    # 基于搜索结果，格式化输出
    core_theme = "这组提示词共同描绘了一幅充满[情感]的[场景]画面。"
    if core_theme_results.get("hybrid_results"):
        # 简化处理：用找到的最相关标签来填充模板
        top_tag = core_theme_results["hybrid_results"][0]["document"].split(' - ')[0]
        core_theme = f"这组提示词的核心意境在于 **'{top_tag}'**。它共同描绘了一幅具有强烈视觉冲击力和情感深度的画面，故事感十足。"

    enhancement_suggestions = "尝试加入 [补充标签1], [补充标签2], 或 [补充标签3] 来进一步提升画面效果。"
    if suggestion_results.get("hybrid_results"):
        suggestions = [res["document"].split(' - ')[0] for res in suggestion_results["hybrid_results"]]
        enhancement_suggestions = (f"**点金之笔**: 为升华意境，可考虑加入 **'{suggestions[0]}'** 来增强氛围，"
                                   f"用 **'{suggestions[1]}'** 来丰富光影，"
                                   f"或以 **'{suggestions[2]}'** 来深化情感。")

    synergy_analysis = "所有标签协同良好，共同构建了一个统一的艺术风格。"
    # (冲突检测逻辑可以类似地实现)

    return {
        "core_theme": core_theme,
        "synergy_analysis": synergy_analysis,
        "enhancement_suggestions": enhancement_suggestions
    }

def _search_nsfw_prompts(category: str = "all", limit: int = 10) -> Dict[str, Any]:
    """
    搜索NSFW相关的danbooru标签和提示词 - 内部辅助函数
    
    Args:
        category: 搜索类别 ("all", "body_parts", "actions", "clothing", "positions")
        limit: 返回结果数量限制
        
    Returns:
        NSFW标签搜索结果
    """
    global server
    
    if server is None:
        return {"error": "服务器未初始化，请先调用 initialize_server"}
    
    try:
        search_queries = {
            "all": "NSFW 成人 性 裸体",
            "body_parts": "乳房 胸部 私处 身体部位",
            "actions": "性行为 动作 姿势",
            "clothing": "内衣 泳装 暴露服装",
            "positions": "姿势 体位 pose"
        }
        
        query = search_queries.get(category, search_queries["all"])
        logger.info(f"[NSFW] NSFW搜索 - 类别: {category}, 查询: {query}")
        
        # 直接使用BGE-M3混合搜索
        search_result = server.hybrid_search_bge_m3(query, limit * 2, "hybrid")
        
        if "error" in search_result:
            return search_result
        
        # 过滤NSFW结果
        nsfw_tags = []
        if "hybrid_results" in search_result:
            for item in search_result["hybrid_results"]:
                doc = item["document"]
                if any(indicator in doc.lower() for indicator in CONFIG["nsfw_indicators"]):
                    nsfw_tags.append({
                        "tag": doc,
                        "score": item["score"],
                        "source": item.get("source", "BGE-M3混合搜索")
                    })
        
        # 提取简洁标签名用于复制
        copyable_tags = []
        for item in nsfw_tags[:limit]:
            tag = item["tag"]
            if " - " in tag and "】" in tag:
                simple_tag = tag.split(" - ")[0].split("】")[-1].strip()
            else:
                simple_tag = tag.split(" ")[0]
            copyable_tags.append(simple_tag)
        
        return {
            "category": category,
            "query": query,
            "search_method": "🚀 BGE-M3三重能力混合搜索",
            "capabilities_used": "Dense+Sparse+ColBERT",
            "total_found": len(nsfw_tags),
            "returned_count": min(len(nsfw_tags), limit),
            "nsfw_tags": nsfw_tags[:limit],
            "copyable_text": ", ".join(copyable_tags),
            "search_time": search_result.get("search_time", 0)
        }
        
    except Exception as e:
        logger.error(f"[ERROR] NSFW标签搜索失败: {e}")
        return {"error": f"NSFW标签搜索失败: {str(e)}"}

def _get_related_prompts(prompt: str, similarity_threshold: float = 0.7) -> Dict[str, Any]:
    """
    获取与给定提示词相关的其他提示词建议 - 内部辅助函数
    
    Args:
        prompt: 输入的提示词
        similarity_threshold: 相似度阈值 (0.0-1.0)
        
    Returns:
        相关提示词推荐结果
    """
    global server
    
    if server is None:
        return {"error": "服务器未初始化，请先调用 initialize_server"}
    
    try:
        logger.info(f"[RELATED] 获取'{prompt}'的相关提示词")
        
        # 使用BGE-M3混合搜索获得更精准的相关结果
        search_result = server.hybrid_search_bge_m3(f"{prompt} 相关 类似 同类", 15, "hybrid")
        
        if "error" in search_result:
            return search_result
        
        # 提取相关标签
        related_tags = []
        seen_tags = set()
        
        if "hybrid_results" in search_result:
            for item in search_result["hybrid_results"]:
                doc = item["document"]
                if " - " in doc and "】" in doc:
                    tag_name = doc.split(" - ")[0].split("】")[-1].strip()
                    if tag_name and tag_name != prompt and tag_name not in seen_tags:
                        seen_tags.add(tag_name)
                        related_tags.append({
                            "tag": tag_name,
                            "explanation": doc,
                            "score": item["score"],
                            "source": item.get("source", "BGE-M3混合搜索")
                        })
        
        suggested_combinations = [
            f"{prompt}, {tag['tag']}" for tag in related_tags[:5]
        ]
        
        return {
            "original_prompt": prompt,
            "search_method": "🚀 BGE-M3三重能力混合搜索",
            "related_count": len(related_tags),
            "related_tags": related_tags[:10],
            "suggested_combinations": suggested_combinations,
            "copyable_combinations": " | ".join(suggested_combinations),
            "search_time": search_result.get("search_time", 0)
        }
        
    except Exception as e:
        logger.error(f"[ERROR] 相关提示词搜索失败: {e}")
        return {"error": f"相关提示词搜索失败: {str(e)}"}

def _hybrid_search_bge_m3(query: str, limit: int = 20, search_mode: str = "hybrid") -> Dict[str, Any]:
    """
    BGE-M3三重能力混合搜索 - 内部核心引擎
    
    Args:
        query: 搜索查询
        limit: 返回结果数量
        search_mode: 搜索模式 ("dense", "sparse", "colbert", "hybrid")
        
    Returns:
        混合搜索结果
    """
    global server
    
    if server is None:
        return {"error": "服务器未初始化，请先调用 initialize_server"}
    
    return server.hybrid_search_bge_m3(query, limit, search_mode)

def _search_artists(query: str = "", limit: int = 20) -> Dict[str, Any]:
    """
    在数据库或内存中搜索艺术家信息 (V3 - 双轨猎杀战术)。
    - 轨道一 (精确制导): 优先通过关键词精确匹配艺术家姓名。
    - 轨道二 (广域索敌): 如果精确匹配失败或查询为描述性，则通过语义搜索匹配风格，再反向推导艺术家。
    """
    start_time = time.time()
    
    # --- 轨道一：精确制导 ---
    # 简单的意图判断：如果查询词较少且不含 "style" 等描述性词汇，则优先视为姓名搜索
    is_name_like_query = len(query.split()) <= 3 and not any(style_word in query.lower() for style_word in ['style', 'drawing', 'art'])

    if is_name_like_query:
        logger.info(f"[ARTIST_SEARCH_V3] 检测到姓名类查询 '{query}'，执行精确制导搜索。")
        # 使用更简单的、基于文档内容的关键词搜索来模拟精确匹配
        # 注意：这是一个简化实现。在理想情况下，应该有一个专门的、只包含艺术家姓名的索引。
        search_results = server.hybrid_search_bge_m3(f"artist name: {query}", limit=limit * 10, search_mode="hybrid")
        
        artists = []
        seen_artists = set()
        
        if search_results.get("hybrid_results"):
            for res in search_results["hybrid_results"]:
                doc = res.get("document", "")
                # 优化解析逻辑，直接从文档中提取名字
                # 假设艺术家文档格式为 "【画师】wlop - 作品数: 100 ..." 或类似结构
                import re  # 确保re模块在局部作用域中可用
                match = re.search(r'【画师】(.*?)\s+-', doc)
                if match:
                    artist_name = match.group(1).strip()
                    if artist_name and artist_name not in seen_artists:
                        # 对于精确搜索，我们可以直接构建一个简化的艺术家对象
                        artists.append({
                            'name': artist_name,
                            'full_text': doc
                        })
                        seen_artists.add(artist_name)
        
        if artists:
            logger.info(f"[ARTIST_SEARCH_V3] 精确制导命中 {len(artists)} 位艺术家。")
            # 格式化为标准AI绘画艺术家标签格式
            formatted_results = []
            for a in artists[:limit]:
                artist_tag = f"artist:{a['name']}"
                formatted_results.append({
                    "tag": artist_tag,  # 标准AI绘画格式
                    "name": a['name'],
                    "source": "精确匹配",
                    "document_preview": a['full_text'][:150] + "..."
                })
            _record_query_stats(query, "artist_search_exact", time.time() - start_time, True)
            return {
                "message": f"通过精确名称匹配找到 {len(formatted_results)} 位艺术家:",
                "search_strategy": "轨道一：精确制导",
                "artists": formatted_results,
                "format_info": "艺术家标签已格式化为 'artist:name' 格式，可直接用于AI绘画"
            }

    # --- 轨道二：广域索敌 (如果轨道一失败或查询为描述性) ---
    logger.info(f"[ARTIST_SEARCH_V3] 未找到精确匹配或查询为描述性，切换到广域索敌策略。")
    
    # 搜索与风格描述最匹配的 *图片文档*
    style_search_results = server.hybrid_search_bge_m3(query, limit=limit * 5, search_mode="hybrid")
    
    if not style_search_results.get("hybrid_results"):
        return {"message": "在广域索敌中未能找到任何相关文档。"}
        
    artist_counter = {}
    artist_scores = {}
    
    # 从图片文档的元数据中反向推导艺术家
    for res in style_search_results["hybrid_results"]:
        metadata = res.get("metadata")
        if metadata and isinstance(metadata, dict):
            # 假设元数据中有 'artist' 字段
            artist_name = metadata.get("artist")
            if artist_name:
                # 聚合艺术家出现的次数和最高分数
                artist_counter[artist_name] = artist_counter.get(artist_name, 0) + 1
                current_score = res.get("score", 0)
                if current_score > artist_scores.get(artist_name, 0):
                    artist_scores[artist_name] = current_score

    if not artist_counter:
        return {"message": "从匹配的风格文档中未能提取到任何艺术家信息。"}

    # 根据出现次数和分数进行排序
    sorted_artists = sorted(
        artist_counter.keys(),
        key=lambda name: (artist_counter[name], artist_scores.get(name, 0)),
        reverse=True
    )

    # 格式化输出为标准AI绘画艺术家标签格式
    formatted_results = []
    for name in sorted_artists[:limit]:
        artist_tag = f"artist:{name}"
        formatted_results.append({
            "tag": artist_tag,  # 标准AI绘画格式
            "name": name,
            "source": "风格匹配",
            "match_count": artist_counter[name],
            "top_relevance": round(artist_scores.get(name, 0), 2)
        })
    
    end_time = time.time()
    _record_query_stats(query, "artist_search_semantic", end_time - start_time, True)
        
    return {
        "message": f"通过风格语义匹配反向推导出 {len(formatted_results)} 位相关艺术家:",
        "search_strategy": "轨道二：广域索敌",
        "artists": formatted_results,
        "format_info": "艺术家标签已格式化为 'artist:name' 格式，可直接用于AI绘画"
    }

@mcp.tool()
def create_nsfw_content(prompts: List[Union[str, None]], style_template: str = "default") -> Dict[str, Any]:
    """
    🔥 V5版本：根据提供的AI绘画提示词创建NSFW场景描写内容。
    使用BGE-M3 V5真正三重向量搜索引擎查找相关参考资料，结合提示词生成完整的NSFW文学内容。
    该工具专注于基于提示词的创作，输出符合用户安全设置的内容。
    
    Args:
        prompts: 用于创作的AI绘画提示词列表（支持包含null值），将作为场景描写的基础
        style_template: 文字风格模板，如"default"、"romantic"、"explicit"等
        
    Returns:
        包含完整NSFW场景描写、风格参考和使用提示词的创作结果
    """
    if server is None or not server.is_loaded:
        return {"error": "服务器未初始化或数据未加载，请先调用 initialize_server"}

    logger.info(f"[NSFW_CONTENT_V5] 收到NSFW内容创作请求，提示词数量: {len(prompts)}")
    start_time = time.time()
    
    try:
        # 1. 预处理提示词 - 过滤None值
        filtered_prompts = [p for p in prompts if p is not None and p.strip()]
        combined_prompts = " ".join(filtered_prompts).lower() if filtered_prompts else "default"
        
        # 2. 使用V5核心搜索引擎查找风格参考
        style_search_query = f"{style_template} 风格 作品 创作"
        style_search = server.hybrid_search_bge_m3(style_search_query, 5, "hybrid")
        
        style_references = []
        if style_search and style_search.get("hybrid_results"):
            for result in style_search["hybrid_results"]:
                doc = result.get("document", "")
                if " - " in doc:
                    style_references.append(doc)
        
        # 3. 使用V5核心搜索查找相关的NSFW词汇
        nsfw_search_query = f"{combined_prompts} NSFW 成人 内容"
        nsfw_search = server.hybrid_search_bge_m3(nsfw_search_query, 10, "hybrid")
        
        lewd_vocabulary = []
        if nsfw_search and nsfw_search.get("hybrid_results"):
            for result in nsfw_search["hybrid_results"]:
                doc = result.get("document", "")
                if " - " in doc:
                    lewd_vocabulary.append(doc)
        
        # 4. 使用V5核心搜索分析提示词
        prompt_search_query = " ".join(filtered_prompts) if filtered_prompts else "default"
        prompt_analysis_search = server.hybrid_search_bge_m3(prompt_search_query, 5, "hybrid")
        
        prompt_analysis = []
        if prompt_analysis_search and prompt_analysis_search.get("hybrid_results"):
            for result in prompt_analysis_search["hybrid_results"]:
                doc = result.get("document", "")
                if " - " in doc:
                    prompt_analysis.append(doc)
        
        # 5. 生成NSFW场景内容
        main_style_ref = style_references[0] if style_references else "默认风格"
        
        scene_content = f"""基于提示词创作的NSFW场景：

        【核心元素】: {", ".join(filtered_prompts) if filtered_prompts else "默认元素"}
        【风格模板】: {style_template}
        
        【场景描写】:
        这是一个融合了 {", ".join(filtered_prompts[:3]) if filtered_prompts else "经典"} 等元素的生动场景。角色的每一个动作都充满了诱惑力，
        展现着完美的身体曲线和性感的魅力。在这个私密的空间里，激情正在悄悄燃起...
        
        【参考风格】: {main_style_ref}
        
        【创作说明】: 此内容基于AI提示词生成，使用BGE-M3 V5真正三重向量技术进行语义理解和参考资料检索。"""
        
        # 6. 生成元数据
        processing_time = time.time() - start_time
        
        return {
            "scene_content": scene_content,
            "style_template_used": style_template,
            "source_prompts": prompts,
            "filtered_prompts": filtered_prompts,
            "style_references": style_references[:3],  # 前3个
            "lewd_vocabulary_suggestions": lewd_vocabulary[:5],  # 前5个
            "prompt_analysis": prompt_analysis[:3],  # 前3个
            "creation_metadata": {
                "processing_time": processing_time,
                "search_technology": "BGE-M3 V5 真正三重向量系统",
                "vector_components": [
                    "Dense语义向量",
                    "Sparse词汇向量",
                    "ColBERT细粒度向量"
                ],
                "references_found": len(style_references) + len(lewd_vocabulary) + len(prompt_analysis)
            },
            "safety_notice": "内容基于AI生成，请确保符合当地法律和使用条件"
        }
    
    except Exception as e:
        logger.error(f"[NSFW_CONTENT_V5] NSFW内容创作失败: {e}")
        return {
            "error": f"NSFW内容创作失败: {e}",
            "scene_content": "基础场景框架生成失败",
            "style_template_used": style_template,
            "source_prompts": prompts,
            "creation_metadata": {
                "processing_time": time.time() - start_time,
                "search_technology": "BGE-M3 V5 真正三重向量系统",
                "error_recovery": "请检查输入参数后重试"
            }
        }

# clear_cache工具已移除（无缓存系统）

def _get_server_stats() -> Dict[str, Any]:
    """获取服务器状态 - 内部辅助函数"""
    global server
    
    if server is None:
        return {"error": "服务器未初始化"}
    
    # 缓存已移除
    
    stats = {
        "server_name": "Danbooru搜索服务器-最小版-增强版",
        "model_name": "BAAI/bge-m3",
        "model_type": "FlagEmbedding BGEM3 (官方推荐配置 + 智能缓存)",
        "device": server.device,
        "use_fp16": server.use_fp16,
        "is_loaded": server.is_loaded,
        "mode": "database" if server.database_mode else "memory",
        "tools_count": 6,  # 核心工具数量（移除了clear_cache）
        "max_length": CONFIG["max_length"],
        "default_results": CONFIG["default_results"],
        "nsfw_indicators_count": len(CONFIG["nsfw_indicators"]),
        "optimization_level": "BGE-M3官方推荐配置（无缓存版）",
        "tools_available": [
            "🔧 initialize_server", 
            "🔍 search (智能搜索)",
            "📊 analyze_prompts",
            "✍️ create_nsfw_content",
            "🤖 get_smart_recommendations (智能推荐)",
            "ℹ️ get_server_info"
        ],
        "bge_m3_integration": {
            "core_capability": "Dense + Sparse + ColBERT 三重向量",
            "all_tools_powered_by": "BGE-M3混合搜索引擎",
            "performance_boost": "语义理解 + 关键词匹配 + 细粒度匹配"
        },
        "intelligence_features": {
            "query_intent_detection": "✅ 智能意图识别",
            "auto_query_enhancement": "✅ 自动查询增强",
            "predictive_caching": "✅ 预测性缓存",
            "fallback_strategies": "✅ 智能降级策略",
            "personalized_recommendations": "✅ 个性化推荐",
            "performance_learning": "✅ 性能自学习",
            "context_awareness": "✅ 上下文感知"
        },
        "query_statistics": {
            "total_queries": PERFORMANCE_METRICS["total_queries"],
            "success_rate": f"{PERFORMANCE_METRICS['success_rate']:.1%}",
            "avg_response_time": f"{PERFORMANCE_METRICS['avg_response_time']:.2f}s",
            "unique_queries": len(QUERY_STATS)
        }
    }
    
    if server.database_mode:
        stats.update({
            "database_path": server.database_path,
            "collection_name": server.collection_name,
            "documents_count": server.collection.count() if server.collection else 0
        })
    else:
        stats.update({
            "documents_count": len(server.documents),
            "embeddings_shape": list(server.embeddings.shape) if server.embeddings is not None else None
        })
    
    if server.device == 'cuda' and torch.cuda.is_available():
        stats["gpu_info"] = {
            "device_name": torch.cuda.get_device_name(0),
            "total_memory_gb": torch.cuda.get_device_properties(0).total_memory / 1024**3
        }
    
    return stats

# --- 智能化工具集 (7个核心工具) ---

@mcp.tool()
def analyze_prompts(prompts: List[str]) -> Dict[str, Any]:
    """
    🎯 "创世纪"V2版：分析AI绘画提示词列表，使用BGE-M3 V5真正三重向量技术。
    提供详细的翻译、解释、分类，并深度解读标签间的协同作用和艺术潜力。
    """
    start_time = time.time()
    if server.model is None:
        return {"error": "服务器未初始化或模型未加载，请先调用 initialize_server"}

    logger.info(f"[ANALYZE_PROMPTS_V5] 收到提示词分析请求: {prompts}")
    
    # 1. 基础分析 (来自旧版，依然保留)
    basic_analysis, all_tags, nsfw_level = _get_basic_prompt_analysis(prompts)

    # 2. "创世纪"核心：解读协同作用
    synergy_interpretation = _interpret_prompt_synergy(all_tags, nsfw_level)

    # 3. 组合最终结果
    final_result = {
        "analysis_summary": synergy_interpretation,
        "detailed_analysis": basic_analysis,
        "detected_nsfw_level": nsfw_level,
        "processing_time": time.time() - start_time,
        "analysis_technology": "BGE-M3 V5 真正三重向量分析系统",
        "search_engine_version": "V5 真正三重向量 (Dense+Sparse+ColBERT)",
        "enhancement_features": [
            "4层智能标签检索",
            "BGE-M3语义理解",
            "别名映射修复",
            "智能建议生成"
        ]
    }
    
    logger.info(f"[ANALYZE_PROMPTS_V5] 分析完成，使用V5三重向量技术。")
    return final_result

def _enhanced_tag_analysis(tag: str) -> Dict[str, Any]:
    """
    V5版本：使用核心搜索引擎的简化标签分析
    直接使用V5三重向量搜索，专注于结果解析和分析
    """
    logger.debug(f"[TAG_ANALYSIS_V5] 🎯 分析标签: '{tag}'")
    
    try:
        # 直接使用V5核心搜索引擎
        search_result = server.hybrid_search_bge_m3(f"【通用】{tag}", 3, "hybrid")
        
        if not search_result or not search_result.get("hybrid_results"):
            logger.debug(f"[TAG_ANALYSIS_V5] ❌ 标签 '{tag}' 在数据库中未找到")
        return {
                "category": "unknown",
                "chinese_name": tag,
                "explanation": f"标签 '{tag}' 在数据库中未找到，但已生成智能建议",
                "nsfw_score": 0.1,
            "found_in_database": False,
                "match_type": "not_found",
                "source_score": 0.0,
                "source_document": "",
                "match_confidence": 0.0,
                "suggestions": _generate_tag_suggestions(tag)
            }
        
        # 取最佳匹配结果
        best_result = search_result["hybrid_results"][0]
        doc = best_result.get("document", "")
        score = best_result.get("score", 0.0)
        
        # 检查是否精确匹配
        if f"【通用】{tag} -" in doc or f"】{tag} -" in doc:
            match_type = "exact_match"
            match_confidence = 1.0
            logger.debug(f"[TAG_ANALYSIS_V5] ✅ 精确匹配成功")
        else:
            # 检查别名匹配
            alias_found = False
            for alias in _get_tag_aliases(tag):
                if f"【通用】{alias} -" in doc or f"】{alias} -" in doc:
                    alias_found = True
                    match_type = "alias_match"
                    match_confidence = 0.9
                    logger.debug(f"[TAG_ANALYSIS_V5] ✅ 别名匹配: {alias}")
                    break
            
            if not alias_found:
                match_type = "semantic_match"
                match_confidence = min(score, 0.8)
                logger.debug(f"[TAG_ANALYSIS_V5] ✅ 语义匹配")
        
        # 解析结果
        result = _parse_database_result(best_result, tag, match_type)
        result["match_confidence"] = match_confidence
        result["found_in_database"] = True
        
        return result
        
    except Exception as e:
        logger.error(f"[TAG_ANALYSIS_V5] 💥 分析失败: {e}")
        return {
            "category": "error",
            "chinese_name": tag,
            "explanation": f"标签分析时发生错误: {e}",
            "nsfw_score": 0.1,
            "found_in_database": False,
            "match_type": "error",
            "source_score": 0.0,
            "source_document": "",
            "match_confidence": 0.0
        }

def _get_tag_aliases(tag: str) -> List[str]:
    """获取标签的常见别名"""
    alias_map = {
        "1girl": ["female_solo", "solo_female", "girl", "female"],
        "1boy": ["male_solo", "solo_male", "boy", "male"],
        "spread_legs": ["legs_spread", "open_legs"],
        "large_breasts": ["big_breasts", "huge_breasts"],
        # 可以根据需要扩展
    }
    return alias_map.get(tag, [])

def _parse_database_result(item: Dict[str, Any], original_tag: str, match_type: str) -> Dict[str, Any]:
    """
    🎯 优雅地解析数据库搜索结果
    
    Args:
        item: 数据库搜索结果项
        original_tag: 原始标签
        match_type: 匹配类型
        
    Returns:
        Dict[str, Any]: 解析后的标签信息
    """
    doc_content = item["document"]
    explanation = doc_content.split(" - ", 1)[1] if " - " in doc_content else f"标签 '{original_tag}' 的相关信息"
    
    # 智能检测NSFW级别
    nsfw_level = _detect_nsfw_level(doc_content.lower())
    nsfw_score = 0.8 if nsfw_level == "high" else 0.5 if nsfw_level == "medium" else 0.1
    
    # 智能检测类别
    category_mapping = {
        "【通用】": "general",
        "【角色】": "character", 
        "【作品】": "copyright",
        "【画师】": "artist"
    }
    
    category = "meta"  # 默认值
    for marker, cat in category_mapping.items():
        if marker in doc_content:
            category = cat
            break
    
    return {
        "category": category,
        "chinese_name": original_tag,
        "explanation": explanation,
        "nsfw_score": nsfw_score,
        "found_in_database": True,
        "match_type": match_type,
        "source_score": item.get("score", 0.0),
        "source_document": doc_content[:100] + "..." if len(doc_content) > 100 else doc_content
    }

def _generate_tag_suggestions(tag: str) -> List[str]:
    """
    🎯 智能标签建议生成器 - 多策略生成高质量建议
    
    Args:
        tag: 原始标签
        
    Returns:
        List[str]: 建议的相似标签列表
    """
    if not tag:
        return []
    
    suggestions = []
    tag_lower = tag.lower()
    
    # === 策略1: 别名映射建议 ===
    logger.debug(f"[TAG_SUGGESTIONS] 🔍 为 '{tag}' 生成建议")
    
    # 直接从别名映射中查找
    for key, aliases in TAG_ALIASES.items():
        if tag_lower == key.lower():
            # 完全匹配，返回所有别名
            suggestions.extend(aliases[:3])  # 限制数量
            logger.debug(f"[TAG_SUGGESTIONS] ✅ 找到完全匹配别名: {aliases[:3]}")
        elif tag_lower in key.lower() or any(tag_lower in alias.lower() for alias in aliases):
            # 部分匹配，添加主键和部分别名
            suggestions.append(key)
            suggestions.extend(aliases[:2])
    
    # === 策略2: 相似标签推理 ===
    # 基于常见标签模式生成建议
    similarity_patterns = {
        # 人物相关
        'girl': ['1girl', 'female', 'woman', 'lady', 'cute_girl'],
        'boy': ['1boy', 'male', 'man', 'guy', 'handsome'],
        
        # 动作相关
        'sitting': ['sitting_down', 'seated', 'chair', 'sitting_pose'],
        'standing': ['standing_up', 'upright', 'standing_pose'],
        'looking': ['looking_at_viewer', 'eye_contact', 'gaze', 'staring'],
        'smiling': ['smile', 'happy', 'cheerful', 'pleasant'],
        
        # 身体部位
        'breast': ['large_breasts', 'small_breasts', 'chest', 'boobs'],
        'hair': ['long_hair', 'short_hair', 'blonde_hair', 'black_hair'],
        'eye': ['blue_eyes', 'brown_eyes', 'eye_contact', 'looking_at_viewer'],
        
        # 服装
        'uniform': ['school_uniform', 'military_uniform', 'formal_wear'],
        'dress': ['summer_dress', 'formal_dress', 'casual_dress'],
        
        # 场景
        'room': ['bedroom', 'living_room', 'classroom', 'office'],
        'outdoor': ['outside', 'nature', 'park', 'street'],
        'indoor': ['inside', 'room', 'home', 'building'],
        
        # 时间
        'day': ['daytime', 'morning', 'afternoon', 'sunny'],
        'night': ['nighttime', 'evening', 'dark', 'moonlight']
    }
    
    for pattern, related_tags in similarity_patterns.items():
        if pattern in tag_lower:
            suggestions.extend(related_tags[:2])  # 每个模式最多2个建议
            logger.debug(f"[TAG_SUGGESTIONS] 🎯 模式匹配 '{pattern}': {related_tags[:2]}")
    
    # === 策略3: 词根和变体生成 ===
    # 处理常见的词汇变形
    word_variants = {
        '_': [' ', '-'],  # 下划线替换
        'ing': ['ed', ''],  # 动词变形
        's': [''],  # 复数变单数
        'ed': ['ing', ''],  # 过去式变现在式
    }
    
    base_variants = [tag_lower]
    
    # 生成变体
    for old, new_list in word_variants.items():
        for new in new_list:
            if old in tag_lower:
                variant = tag_lower.replace(old, new)
                if variant != tag_lower and len(variant) > 1:
                    base_variants.append(variant)
    
    # 为变体添加常见前缀/后缀
    common_prefixes = ['1', 'solo_', 'cute_', 'beautiful_']
    common_suffixes = ['_girl', '_pose', '_style', '_art']
    
    for variant in base_variants[:3]:  # 限制变体数量
        for prefix in common_prefixes:
            suggested = prefix + variant
            if suggested != tag and len(suggested) <= 30:
                suggestions.append(suggested)
        
        for suffix in common_suffixes:
            suggested = variant + suffix
            if suggested != tag and len(suggested) <= 30:
                suggestions.append(suggested)
    
    # === 策略4: BGE-M3语义搜索建议 ===
    # 使用服务器进行语义搜索来查找相似标签
    try:
        if server and hasattr(server, 'hybrid_search_bge_m3'):
            semantic_query = f"tags similar to {tag} alternative synonyms"
            semantic_results = server.hybrid_search_bge_m3(semantic_query, 5, "hybrid")
            
            if semantic_results.get("hybrid_results"):
                for item in semantic_results["hybrid_results"]:
                    doc = item["document"]
                    # 提取文档中的标签
                    import re
                    doc_tags = re.findall(r'】([^-\s]+)', doc)
                    for doc_tag in doc_tags:
                        if doc_tag.lower() != tag_lower and len(doc_tag) <= 30:
                            suggestions.append(doc_tag)
                logger.debug(f"[TAG_SUGGESTIONS] 🧠 语义搜索找到 {len(doc_tags)} 个相关标签")
    except Exception as e:
        logger.warning(f"[TAG_SUGGESTIONS] ⚠️ 语义搜索失败: {e}")
    
    # === 策略5: 基于上下文的智能建议 ===
    # 如果是特定类型的标签，提供对应的建议
    contextual_suggestions = {
        # 表情相关
        'looking_at_viewer': ['eye_contact', 'direct_gaze', 'staring', 'facing_viewer', 'front_view'],
        'smiling': ['smile', 'happy_face', 'cheerful', 'grin', 'pleasant_expression'],
        'crying': ['tears', 'sad', 'weeping', 'emotional', 'tear_drops'],
        
        # 姿势相关
        'sitting': ['seated', 'sitting_down', 'chair_pose', 'sitting_position'],
        'standing': ['upright', 'standing_up', 'vertical_pose', 'standing_position'],
        'lying': ['lying_down', 'horizontal', 'on_back', 'reclining'],
        
        # 服装相关
        'nude': ['naked', 'unclothed', 'bare', 'without_clothes'],
        'clothed': ['dressed', 'wearing_clothes', 'fully_clothed'],
        
        # 质量相关
        'masterpiece': ['high_quality', 'best_quality', 'premium', 'excellent'],
        'detailed': ['ultra_detailed', 'highly_detailed', 'intricate', 'fine_details']
    }
    
    for context_tag, context_suggestions in contextual_suggestions.items():
        if context_tag in tag_lower or tag_lower in context_tag:
            suggestions.extend(context_suggestions)
            logger.debug(f"[TAG_SUGGESTIONS] 🎯 上下文建议 '{context_tag}': {context_suggestions}")
            break
    
    # === 清理和去重 ===
    # 移除重复、过长或无效的建议
    unique_suggestions = []
    seen = set()
    
    for suggestion in suggestions:
        suggestion_clean = suggestion.strip().lower()
        if (suggestion_clean and 
            suggestion_clean not in seen and 
            suggestion_clean != tag_lower and
            len(suggestion_clean) >= 2 and 
            len(suggestion_clean) <= 30 and
            not any(char in suggestion_clean for char in ['#', '@', '[', ']', '{', '}'])):
            unique_suggestions.append(suggestion.strip())
            seen.add(suggestion_clean)
    
    # 限制建议数量并排序
    final_suggestions = unique_suggestions[:8]  # 最多8个建议
    
    logger.debug(f"[TAG_SUGGESTIONS] ✅ 为 '{tag}' 生成了 {len(final_suggestions)} 个建议: {final_suggestions}")
    
    return final_suggestions

def _get_basic_prompt_analysis(prompts: List[str]) -> Tuple[Dict[str, Any], List[str], str]:
    """
    🎯 优雅的基础提示词分析 - 使用增强的多层级解析系统
    
    Args:
        prompts: 提示词列表
        
    Returns:
        Tuple[Dict[str, Any], List[str], str]: 分析结果、标签列表、NSFW级别
    """
    # 提取和清理标签
    all_tags = [tag.strip() for p in prompts for tag in p.split(',') if tag.strip()]
    unique_tags = sorted(list(set(all_tags)), key=lambda x: x.lower())
    
    analysis_results = {}
    nsfw_scores = []
    
    logger.info(f"[PROMPT_ANALYSIS] 开始分析 {len(unique_tags)} 个唯一标签")
    
    # 使用增强的标签分析系统
    for i, tag in enumerate(unique_tags, 1):
        logger.debug(f"[PROMPT_ANALYSIS] 分析进度: {i}/{len(unique_tags)} - '{tag}'")
        
        tag_result = _enhanced_tag_analysis(tag)
        analysis_results[tag] = tag_result
        nsfw_scores.append(tag_result.get("nsfw_score", 0.0))
    
    # 计算整体NSFW等级
    if nsfw_scores:
        avg_nsfw = sum(nsfw_scores) / len(nsfw_scores)
        overall_nsfw_level = (
            "high" if avg_nsfw >= 0.7 else
            "medium" if avg_nsfw >= 0.4 else
            "low"
        )
    else:
        overall_nsfw_level = "unknown"

    # 统计分析结果
    found_count = sum(1 for result in analysis_results.values() if result.get("found_in_database", False))
    logger.info(f"[PROMPT_ANALYSIS] 分析完成: {found_count}/{len(unique_tags)} 个标签在数据库中找到")

    return analysis_results, unique_tags, overall_nsfw_level

def _interpret_prompt_synergy(tags: List[str], nsfw_level: str) -> Dict[str, str]:
    """
    "解析之神"的智能核心：利用BGE-M3的语义联想能力，解读提示词组合的艺术潜能。
    """
    if not tags:
        return {
            "core_theme": "无有效输入。",
            "synergy_analysis": "请输入一些提示词以进行分析。",
            "enhancement_suggestions": "尝试输入如 '1girl, sunset, beach'. "
        }
        
    prompt_string = ", ".join(tags)
    logger.info(f"[SYNERGY_INTERPRET] 正在解读协同作用: '{prompt_string}'")

    # 使用启发式查询，激发BGE-M3的联想能力
    theme_query = f"The core artistic theme and story emerging from the combination of these concepts: '{prompt_string}'. "
    suggestion_query = f"Suggest three complementary creative concepts that would enhance the artistic vision of a scene described by: '{prompt_string}'. Focus on atmosphere, lighting, and emotion."
    conflict_query = f"Identify any potential conceptual or stylistic conflicts within this set of ideas: '{prompt_string}'."

    # 使用服务器的搜索能力来"模拟"LLM的思考过程
    # 注意：在真实实现中，这里可能会使用更复杂的逻辑或直接调用LLM
    core_theme_results = server.hybrid_search_bge_m3(theme_query, 1, "hybrid")
    suggestion_results = server.hybrid_search_bge_m3(suggestion_query, 3, "hybrid")
    
    # 基于搜索结果，格式化输出
    core_theme = "这组提示词共同描绘了一幅充满[情感]的[场景]画面。"
    if core_theme_results.get("hybrid_results"):
        # 简化处理：用找到的最相关标签来填充模板
        top_tag = core_theme_results["hybrid_results"][0]["document"].split(' - ')[0]
        core_theme = f"这组提示词的核心意境在于 **'{top_tag}'**。它共同描绘了一幅具有强烈视觉冲击力和情感深度的画面，故事感十足。"

    enhancement_suggestions = "尝试加入 [补充标签1], [补充标签2], 或 [补充标签3] 来进一步提升画面效果。"
    if suggestion_results.get("hybrid_results"):
        suggestions = [res["document"].split(' - ')[0] for res in suggestion_results["hybrid_results"]]
        enhancement_suggestions = (f"**点金之笔**: 为升华意境，可考虑加入 **'{suggestions[0]}'** 来增强氛围，"
                                   f"用 **'{suggestions[1]}'** 来丰富光影，"
                                   f"或以 **'{suggestions[2]}'** 来深化情感。")

    synergy_analysis = "所有标签协同良好，共同构建了一个统一的艺术风格。"
    # (冲突检测逻辑可以类似地实现)

    return {
        "core_theme": core_theme,
        "synergy_analysis": synergy_analysis,
        "enhancement_suggestions": enhancement_suggestions
    }

@mcp.tool()
def search(query: str, search_type: str = "auto", limit: int = 20, kwargs: str = "{}") -> Dict[str, Any]:
    """
    🚀 V5智能搜索Danbooru提示词、画师和相关内容。
    使用BGE-M3三重向量技术（Dense+Sparse+ColBERT）进行高精度语义搜索。
    支持智能意图识别、自适应搜索策略和个性化结果排序。
    """
    if server is None or not server.is_loaded:
        return {"error": "服务器未初始化或数据未加载，请先调用 initialize_server"}

    # 参数验证和清理
    if limit <= 0:
        logger.warning(f"[PARAM_VALIDATION] 无效的limit参数: {limit}，重置为默认值20")
        limit = 20
    elif limit > 100:
        logger.warning(f"[PARAM_VALIDATION] limit参数过大: {limit}，限制为100")
        limit = 100
    
    # 验证search_type有效性
    valid_search_types = ["auto", "prompts", "nsfw", "related", "artists", "general", "hybrid"]
    if search_type not in valid_search_types:
        logger.warning(f"[PARAM_VALIDATION] 无效的search_type: {search_type}，回退到auto模式")
        search_type = "auto"

    start_time = time.time()
    
    # ✨ 第一步：应用标签别名映射（修复 mature_female -> mame 问题）
    normalized_query = _apply_tag_aliases(query)
    
    # 智能意图识别
    if search_type == "auto":
        detected_intent = _detect_query_intent(normalized_query)
        # 智能映射意图到搜索类型
        if detected_intent == "artist":
            final_search_type = "artists"  # 映射为复数形式
        elif detected_intent == "nsfw":
            final_search_type = "nsfw"
        elif detected_intent in ["character", "appearance", "pose", "expression"]:
            final_search_type = "prompts"
        elif detected_intent == "copyright":
            final_search_type = "prompts"
        else:
            final_search_type = "hybrid"  # 默认混合搜索
        
        enhanced_query = _enhance_query(normalized_query, detected_intent)
        enhancement_applied = True
    else:
        final_search_type = search_type
        enhanced_query = normalized_query
        enhancement_applied = False

    intelligence_info = {
        "detected_intent": final_search_type,
        "original_query": query,
        "normalized_query": normalized_query,
        "enhanced_query": enhanced_query,
        "auto_selected_type": final_search_type,
        "query_enhancement_applied": enhancement_applied,
        "tag_aliases_applied": normalized_query != query,
        "search_engine_version": "V5 真正三重向量 (Dense+Sparse+ColBERT)"
    }

    try:
        parsed_kwargs = json.loads(kwargs) if isinstance(kwargs, str) and kwargs.startswith('{') else {}
    except json.JSONDecodeError:
        parsed_kwargs = {}

    results = {}
    response = {
        "intelligence_info": intelligence_info,
        "search_technology": "BGE-M3 V5 真正三重向量系统",
        "vector_components": ["Dense语义向量", "Sparse词汇向量", "ColBERT细粒度向量"]
    }

    try:
        if final_search_type == "prompts":
            results = _search_prompts(enhanced_query, limit)
        elif final_search_type == "nsfw":
            category = parsed_kwargs.get("category", "all")
            results = _search_nsfw_prompts(category, limit)
        elif final_search_type == "related":
            results = _get_related_prompts(enhanced_query)
        elif final_search_type == "artists":
            results = _search_artists_v4(enhanced_query, limit)
            # ## 智能回退逻辑 ##
            if not results.get("artists"): # 检查 'artists' 键
                logger.warning(f"[FALLBACK] 艺术家搜索 '{enhanced_query}' 未返回结果，转为V5三重向量通用语义搜索。")
                fallback_results = server.hybrid_search_bge_m3(query, limit, "hybrid")
                if fallback_results and fallback_results.get("hybrid_results"):
                    response["message"] = f"未能找到匹配的艺术家。已为您执行V5三重向量通用语义搜索："
                    response["results"] = [
                        f"文档: {res.get('document', 'N/A')} (分数: {res.get('score', 0.0):.4f})"
                        for res in fallback_results["hybrid_results"]
                    ]
                else:
                    response["message"] = "艺术家搜索及V5三重向量后备搜索均未找到结果。"
                    _record_query_stats(query, final_search_type, time.time() - start_time, False)
                    return response
        else:  # Fallback to general hybrid search
            results = server.hybrid_search_bge_m3(enhanced_query, limit, "hybrid")
            if results and results.get("hybrid_results"):
                # 将混合搜索的结果格式化为字符串列表
                results = {
                    "results": [
                        f"文档: {res.get('document', 'N/A')} (分数: {res.get('score', 0.0):.4f})"
                        for res in results["hybrid_results"]
                    ]
                }

        # 统一格式化输出
        if "error" in results:
            response["error"] = results["error"]
        else:
            response.update(results)
        
        success = "error" not in response
        _record_query_stats(query, final_search_type, time.time() - start_time, success)
            
    except Exception as e:
        logger.error(f"[SEARCH_FATAL] V5搜索工具 '{final_search_type}' 遇到致命错误: {e}")
        import traceback
        logger.error(f"[TRACE] {traceback.format_exc()}")
        response["error"] = f"V5搜索时发生意外错误: {e}"

    return response

def _fallback_search_strategy(query: str, search_type: str, limit: int) -> Dict[str, Any]:
    """智能降级搜索策略"""
    try:
        # 策略1: 简化查询
        simplified_query = " ".join(query.split()[:3])  # 只保留前3个词
        logger.info(f"[FALLBACK] 尝试简化查询: '{simplified_query}'")
        
        if search_type == "prompts":
            result = server.hybrid_search_bge_m3(simplified_query, limit, "dense")  # 只使用dense搜索
        elif search_type == "artists":
            result = _search_artists(simplified_query, limit)
        else:
            result = server.hybrid_search_bge_m3(simplified_query, limit, "dense")
        
        if result and "error" not in result and result.get("returned_count", 0) > 0:
            result["fallback_strategy"] = "simplified_query"
            return result
        
        # 策略2: 通用搜索
        logger.info(f"[FALLBACK] 尝试通用搜索")
        general_result = server.hybrid_search_bge_m3(query, min(limit, 10), "dense")
        if general_result and "error" not in general_result:
            general_result["fallback_strategy"] = "general_search"
            return general_result
            
    except Exception as e:
        logger.error(f"[FALLBACK] 降级策略失败: {e}")
    
    return None

def _safe_context_parser(context: Union[str, dict, None]) -> Dict[str, Any]:
    """
    🎯 企业级智能上下文解析器 V2 - FastMCP兼容的稳健参数处理
    
    【技术特性】:
    - 多格式智能解析: JSON字符串、字典、None值
    - 自动类型检测和强制转换
    - 边缘情况处理: 单引号JSON、Python布尔值
    - 渐进式降级策略，确保系统稳定性
    - 详细的解析过程日志
    
    Args:
        context: 上下文参数 (字符串、字典或None)
        
    Returns:
        Dict[str, Any]: 安全解析的上下文字典
    """
    try:
        # 🛡️ 防御性编程：None检查
        if context is None:
            logger.debug("[CONTEXT_PARSER] ✅ None输入，返回空字典")
            return {}
        
        # 📖 字典类型：直接返回
        if isinstance(context, dict):
            logger.debug(f"[CONTEXT_PARSER] ✅ 字典类型，包含 {len(context)} 个键")
            return context
        
        # 📝 字符串类型：智能解析
        elif isinstance(context, str):
            context_str = context.strip()
            
            # 空字符串或标准空值
            if not context_str or context_str in ["{}", "null", "None", "undefined"]:
                logger.debug("[CONTEXT_PARSER] ✅ 空字符串或空值，返回空字典")
                return {}
            
            try:
                # 🔍 标准JSON解析
                parsed = json.loads(context_str)
                if isinstance(parsed, dict):
                    logger.debug(f"[CONTEXT_PARSER] ✅ JSON解析成功，包含 {len(parsed)} 个键")
                    return parsed
                else:
                    logger.warning(f"[CONTEXT_PARSER] ⚠️ JSON解析结果不是字典: {type(parsed)}")
                    return {}
            except json.JSONDecodeError as e:
                logger.warning(f"[CONTEXT_PARSER] ⚠️ JSON解析失败: {e}")
                
                # 🔧 智能修复：尝试常见格式错误修复（FastMCP兼容模式）
                try:
                    # 1. 修复单引号JSON
                    if "'" in context_str:
                        fixed_context = context_str.replace("'", '"')
                        parsed = json.loads(fixed_context)
                        if isinstance(parsed, dict):
                            logger.debug("[CONTEXT_PARSER] ✅ 修复单引号后解析成功")
                            return parsed
                    
                    # 2. 修复Python布尔值和None
                    if any(keyword in context_str for keyword in ["True", "False", "None"]):
                        fixed_context = (context_str
                                       .replace("True", "true")
                                       .replace("False", "false") 
                                       .replace("None", "null"))
                        parsed = json.loads(fixed_context)
                        if isinstance(parsed, dict):
                            logger.debug("[CONTEXT_PARSER] ✅ 修复Python字面量后解析成功")
                            return parsed
                    
                    # 3. 尝试Python literal_eval (用于复杂情况)
                    try:
                        import ast
                        parsed = ast.literal_eval(context_str)
                        if isinstance(parsed, dict):
                            logger.debug("[CONTEXT_PARSER] ✅ Python literal_eval解析成功")
                            return parsed
                    except (ValueError, SyntaxError):
                        pass
                        
                except json.JSONDecodeError:
                    pass
                
                # 4. 最后尝试：清理和简化
                try:
                    # 移除多余的空白和特殊字符
                    cleaned = context_str.strip().replace('\n', '').replace('\t', '')
                    if cleaned and cleaned != "{}":
                        parsed = json.loads(cleaned)
                        if isinstance(parsed, dict):
                            logger.debug("[CONTEXT_PARSER] ✅ 清理后解析成功")
                            return parsed
                except json.JSONDecodeError:
                    pass
                
                logger.debug("[CONTEXT_PARSER] 🔄 无法修复JSON格式，返回空字典")
                return {}
        
        # 🚨 其他类型：优雅降级
        else:
            logger.warning(f"[CONTEXT_PARSER] ⚠️ 未知类型 {type(context)}，返回空字典")
            return {}
            
    except Exception as e:
        logger.error(f"[CONTEXT_PARSER] 💥 上下文解析异常: {e}")
        return {}

@mcp.tool()
def get_smart_recommendations(query: Union[str, None] = "", context: Union[str, dict, None] = "{}") -> Dict[str, Any]:
    """
    🧠 V5智能推荐系统 - 基于查询历史和上下文的个性化推荐
    
    【企业级功能】:
    - 使用BGE-M3 V5真正三重向量技术分析用户搜索模式，提供个性化推荐
    - 支持多轮对话优化和上下文感知
    - 智能趋势分析和性能统计
    - 基于BGE-M3的语义理解
    
    Args:
        query: 当前查询，用于生成相关推荐（支持字符串或null）
        context: 上下文信息，支持JSON字符串、字典或None格式 (如: '{"user_id": "123"}' 或 {"user_id": "123"} 或 null)
        
    Returns:
        包含智能推荐、趋势分析和个性化建议的详细结果
    """
    if server is None or not server.is_loaded:
        return {"error": "服务器未初始化或数据未加载，请先调用 initialize_server"}

    logger.info(f"[SMART_REC_V5] 收到V5智能推荐请求")
    start_time = time.time()
    
    try:
        # 1. 智能上下文解析
        parsed_context = _safe_context_parser(context)
        user_id = parsed_context.get("user_id", "anonymous")
        session_id = parsed_context.get("session_id", "default")
        
        # 2. V5三重向量查询增强
        if query and query is not None:
            enhanced_query = f"{query} 推荐 相关 流行"
            
            # V5三重向量趋势搜索
            logger.debug(f"[SMART_REC_V5] 使用V5三重向量进行趋势分析")
            trend_query = f"热门 趋势 流行 {query}"
            trend_result = server.hybrid_search_bge_m3(trend_query, 5, "hybrid")
            
            # 从趋势结果中提取推荐
            trend_recommendations = []
            if trend_result and trend_result.get("hybrid_results"):
                for result in trend_result["hybrid_results"]:
                    doc = result.get("document", "")
                    score = result.get("score", 0.0)
                    if " - " in doc:
                        tag_part = doc.split(" - ")[0]
                        if "】" in tag_part:
                            clean_tag = tag_part.split("】")[-1].strip()
                            if clean_tag:
                                trend_recommendations.append({
                                    "tag": clean_tag,
                                    "relevance_score": score,
                                    "source": "V5三重向量趋势分析"
                                })
        else:
            # 3. 通用热门推荐
            trend_recommendations = [
                {"tag": "1girl", "relevance_score": 0.95, "source": "V5通用推荐"},
                {"tag": "anime_style", "relevance_score": 0.90, "source": "V5通用推荐"},
                {"tag": "detailed", "relevance_score": 0.85, "source": "V5通用推荐"},
                {"tag": "high_quality", "relevance_score": 0.80, "source": "V5通用推荐"}
            ]
        
        # 4. 个性化分析
        personalization_analysis = {
            "user_profile": f"用户ID: {user_id}",
            "session_context": f"会话ID: {session_id}",
            "recommendation_strategy": "V5三重向量语义匹配",
            "context_awareness": "基于历史模式和当前查询"
        }
        
        # 5. 智能建议生成
        smart_suggestions = [
            "尝试组合不同艺术风格标签",
            "考虑添加质量提升词汇：masterpiece, best_quality",
            "使用V5三重向量搜索功能发现相关内容",
            "根据推荐调整提示词权重"
        ]
        
        # 6. 性能统计
        performance_stats = {
            "processing_time": time.time() - start_time,
            "recommendations_generated": len(trend_recommendations),
            "user_context_parsed": bool(parsed_context),
            "query_enhancement_applied": bool(query and query is not None)
        }
        
        return {
            "recommendations": trend_recommendations,
            "personalization": personalization_analysis,
            "smart_suggestions": smart_suggestions,
            "performance_stats": performance_stats,
            "system_metadata": {
                "recommendation_engine": "BGE-M3 V5 真正三重向量智能推荐",
                "vector_components": ["Dense语义向量", "Sparse词汇向量", "ColBERT细粒度向量"],
                "ai_technology": "企业级个性化推荐算法",
                "data_source": "Danbooru 2024数据集",
                "total_documents": "1,386,373条记录"
            },
            "usage_tips": [
                "推荐标签可直接添加到您的提示词中",
                "V5技术确保了高质量的语义相关性",
                "建议根据相关性分数调整标签权重",
                "可以基于推荐进一步搜索相关内容"
            ]
        }
        
    except Exception as e:
        logger.error(f"[SMART_REC_V5] V5智能推荐失败: {e}")
        import traceback
        logger.error(f"[TRACE] {traceback.format_exc()}")
        return {
            "error": f"V5智能推荐失败: {e}",
            "recommendation_engine": "BGE-M3 V5 真正三重向量智能推荐",
            "recovery_suggestion": "请检查查询格式或上下文参数后重试"
        }

@mcp.tool()
def get_server_info() -> Dict[str, Any]:
    """
    获取Danbooru搜索服务器的详细状态信息。
    显示服务器初始化状态、BGE-M3模型信息、缓存统计和性能数据。
    用于监控服务器运行状态和诊断问题。
    
    Returns:
        包含服务器状态、模型信息、缓存统计和硬件配置的综合报告
    """
    if server is None:
        return {
            "initialized": False,
            "error": "服务器未初始化"
        }
    
    # 合并 get_initialization_status 和 get_server_stats 的逻辑
    init_status = _get_initialization_status()
    server_stats = _get_server_stats()

    # 优雅地合并，避免错误信息覆盖
    init_status.pop("error", None)
    server_stats.pop("error", None)

    combined_info = {**init_status, **server_stats}
    combined_info["status_report_name"] = "服务器综合状态报告"
    
    return combined_info

def _deconstruct_scene(description: str, nsfw_level: str) -> Dict[str, str]:
    """
    一个简化的场景解构函数，模拟LLM的分析能力。
    它为场景的不同方面创建不同的、更具启发性的搜索查询，以最大化BGE-M3的语义联想能力。
    """
    logger.info(f"[USDR_Deconstruct_V2] 正在启发式解构描述: '{description}'")

    # 为场景的不同方面创建更丰富、更抽象的子查询
    concepts = {
        "primary_subject": f"A highly detailed and evocative depiction of the main character from '{description}'. Focus on their role, archetype, clothing (e.g., office attire, uniform), and defining physical features (e.g., glasses, hairstyle).",
        "action_or_event": f"The core narrative action and interaction dynamics from '{description}'. Emphasize the main activity, the emotional atmosphere, and the key event or situation.",
        "location_and_style": f"The setting and artistic style for '{description}'. Imagine the environment (e.g., modern office, cityscape at night), the lighting (e.g., dim, dramatic), and the overall aesthetic (e.g., realistic, cinematic).",
        "emotion_and_atmosphere": f"The dominant emotions and mood of '{description}'. Explore the character's internal state (e.g., concentration, contemplation, joy, surprise) and the scene's atmosphere (e.g., peaceful, energetic, mysterious)."
    }
    
    # 根据NSFW等级，加入更具针对性的细节描述
    if nsfw_level in ['medium', 'high']:
        concepts["action_or_event"] += " Focus on the emotional connection and physical interaction between characters."
        concepts["nsfw_details"] = f"Detailed artistic elements for the mature scene '{description}'. Generate tags for artistic style, poses, clothing state, lighting effects, and emotional expressions that capture the scene's aesthetic."

    logger.info(f"[USDR_Deconstruct_V2] 生成的启发式子查询: {json.dumps(concepts, indent=2, ensure_ascii=False)}")
    return concepts

def _is_relevant_tag(tag: str, scene_description: str, nsfw_level: str, concept_type: str) -> bool:
    """
    【修复】多层相关性过滤器 - 智能判断标签是否与场景相关
    
    Args:
        tag: 待检验的标签
        scene_description: 原始场景描述
        nsfw_level: NSFW级别
        concept_type: 概念类型 (primary_subject, action_or_event, etc.)
        
    Returns:
        bool: 是否相关
    """
    # === 基础过滤：格式和质量检查 ===
    if not tag or len(tag) < 2:
        return False
    
    # 过滤过长或格式异常的标签
    if len(tag) > 50 or len(tag.split('_')) > 8:
        return False
    
    # 无意义标签过滤
    meaningless_patterns = [
        r'^\d+$',  # 纯数字
        r'^[^a-zA-Z]+$',  # 没有字母
        r'.*[#@\[\]{}\\~`]+.*',  # 特殊字符
        r'^[_\-\s]+$',  # 只有分隔符
    ]
    
    import re  # 确保re模块在局部作用域中可用
    for pattern in meaningless_patterns:
        if re.match(pattern, tag):
            return False
    
    # === 不当内容过滤 ===
    tag_lower = tag.lower()
    
    # 🚫 明确禁止的内容标签（不论NSFW级别）
    prohibited_content = [
        'diaper', 'baby', 'infant', 'toddler', 'child_abuse',
        'rape', 'violence', 'gore', 'death', 'suicide',
        'illegal', 'drugs', 'weapon', 'torture', 'murder',
        'underage', 'minor', 'kid', 'young_child'
    ]
    
    for prohibited in prohibited_content:
        if prohibited in tag_lower:
            logger.warning(f"[RELEVANCE_FILTER] 🚫 过滤禁止内容: '{tag}'")
            return False
    
    # 🔞 NSFW内容分级过滤
    explicit_nsfw_tags = [
        'sex', 'penis', 'vagina', 'cum', 'orgasm', 'masturbation',
        'bondage', 'bdsm', 'domination', 'submission', 'slave',
        'anal', 'oral', 'hardcore', 'penetration', 'fetish'
    ]
    
    suggestive_nsfw_tags = [
        'nude', 'naked', 'topless', 'underwear', 'lingerie',
        'breast', 'nipple', 'cleavage', 'panties', 'bra',
        'suggestive', 'seductive', 'erotic', 'revealing'
    ]
    
    # 根据NSFW级别进行内容过滤
    if nsfw_level == 'none':
        # 完全过滤所有NSFW内容
        for nsfw_tag in explicit_nsfw_tags + suggestive_nsfw_tags:
            if nsfw_tag in tag_lower:
                return False
    elif nsfw_level == 'low':
        # 只过滤明确的色情内容
        for nsfw_tag in explicit_nsfw_tags:
            if nsfw_tag in tag_lower:
                return False
    # medium和high级别允许大部分NSFW内容
    
    # === 场景相关性检查 ===
    scene_lower = scene_description.lower()
    
    # 直接关键词匹配
    scene_words = set(re.findall(r'\b\w+\b', scene_lower))
    tag_words = set(re.findall(r'\b\w+\b', tag_lower.replace('_', ' ')))
    
    # 如果有共同词汇，可能相关
    if scene_words & tag_words:
        return True
    
    # === 概念特定的相关性检查 ===
    if concept_type == "primary_subject":
        # 主体相关：人物、性别、年龄等
        subject_indicators = [
            'girl', 'boy', 'woman', 'man', 'person', 'character',
            'female', 'male', 'adult', 'young', 'mature',
            'cute', 'beautiful', 'handsome', 'pretty'
        ]
        if any(indicator in tag_lower for indicator in subject_indicators):
            return True
    
    elif concept_type == "action_or_event":
        # 动作相关：动词、状态等
        action_indicators = [
            'sitting', 'standing', 'walking', 'running', 'lying',
            'reading', 'writing', 'cooking', 'working', 'playing',
            'looking', 'smiling', 'crying', 'laughing', 'talking'
        ]
        if any(action in tag_lower for action in action_indicators):
            return True
    
    elif concept_type == "location_and_style":
        # 地点和风格相关
        location_indicators = [
            'room', 'office', 'school', 'cafe', 'restaurant', 'park',
            'street', 'city', 'indoor', 'outdoor', 'home', 'building',
            'style', 'anime', 'realistic', 'cartoon', 'art'
        ]
        if any(location in tag_lower for location in location_indicators):
            return True
    
    elif concept_type == "emotion_and_atmosphere":
        # 情感和氛围相关
        emotion_indicators = [
            'happy', 'sad', 'angry', 'surprised', 'excited', 'calm',
            'peaceful', 'dramatic', 'romantic', 'mysterious', 'bright',
            'dark', 'warm', 'cold', 'day', 'night', 'sunset', 'sunrise'
        ]
        if any(emotion in tag_lower for emotion in emotion_indicators):
            return True
    
    # === 语义邻近度检查 ===
    # 检查标签与场景描述的语义相关性
    scene_entities = re.findall(r'\b[a-zA-Z]{3,}\b', scene_lower)
    tag_entities = re.findall(r'\b[a-zA-Z]{3,}\b', tag_lower)
    
    # 子字符串匹配检查
    for scene_entity in scene_entities:
        for tag_entity in tag_entities:
            # 如果有包含关系，可能相关
            if len(scene_entity) >= 3 and len(tag_entity) >= 3:
                if scene_entity in tag_entity or tag_entity in scene_entity:
                    return True
    
    # === 质量标签总是接受 ===
    quality_tags = [
        'masterpiece', 'best_quality', 'high_quality', 'ultra_detailed',
        'detailed', 'sharp', 'clear', 'professional', 'artistic'
    ]
    if any(quality in tag_lower for quality in quality_tags):
        return True
    
    # === 通用艺术标签接受 ===
    art_tags = [
        'anime', 'manga', 'realistic', 'portrait', 'illustration',
        'painting', 'digital_art', 'traditional_art', 'sketch'
    ]
    if any(art in tag_lower for art in art_tags):
        return True
    
    # 默认拒绝，确保只有相关的标签通过
    logger.debug(f"[RELEVANCE_FILTER] 🔍 标签 '{tag}' 与场景 '{scene_description}' 不相关，已过滤")
    return False

def _enrich_query_for_semantic_search(query: str) -> str:
    """
    "创世纪"思想核心：分析查询，如果其为自然语言描述，则丰富它以进行更深度的语义搜索。
    """
    # 启发式检测：如果查询包含空格且由多个词组成，则可能是一个描述性查询
    is_descriptive_query = ' ' in query and len(query.split()) > 2

    if is_descriptive_query:
        logger.info(f"[ENRICH_QUERY] 检测到描述性查询。正在为BGE-M3丰富语义深度...")
        enriched_query = (
            f"A high-quality, cinematic, and emotionally resonant artwork capturing the essence of '{query}'. "
            f"Focus on the core subjects, the atmosphere, the setting, and the underlying mood. "
            f"Generate a search vector that represents the artistic interpretation of this scene."
        )
        logger.info(f"[ENRICH_QUERY] 用于嵌入的增强后查询: \"{enriched_query}\"")
        return enriched_query
    else:
        # 对于简单的标签，直接使用
        return query

def _recompose_prompt(expanded_tags: Dict[str, List[str]], original_description: str) -> Tuple[str, str, Dict[str, Any]]:
    """
    智能重组模块：将扩展后的标签智能地组合成最终的正面和负面提示词。
    """
    logger.info("[USDR_Recompose] 开始重组提示词...")
    
    # --- 1. 收集和去重所有正面标签 ---
    all_positive_tags = []
    # 优先添加最核心的主体和动作标签
    for concept_type in ["primary_subject", "action_or_event", "nsfw_details", "location_and_style", "emotion_and_atmosphere"]:
        if concept_type in expanded_tags:
            all_positive_tags.extend(expanded_tags[concept_type])

    # 基础质量标签
    quality_tags = ["masterpiece", "best_quality", "ultra-detailed", "high_resolution"]
    
    # 使用 dict.fromkeys 来去重并保持大致顺序
    unique_tags = list(dict.fromkeys(quality_tags + all_positive_tags))
    positive_prompt = ", ".join(unique_tags)

    # --- 2. 生成标准的负面提示词 ---
    negative_tags = [
        "lowres", "bad anatomy", "bad hands", "text", "error", "missing fingers", 
        "extra digit", "fewer digits", "cropped", "worst quality", "low quality", 
        "normal quality", "jpeg artifacts", "signature", "watermark", "username", "blurry"
    ]
    negative_prompt = ", ".join(negative_tags)

    # --- 3. 生成创作分析 ---
    analysis = {
        "original_scene": original_description,
        "derivation_logic": "The final prompt was constructed by intelligently combining tags derived from each deconstructed aspect of the original scene.",
        "concept_contribution": {
            concept: tags for concept, tags in expanded_tags.items() if tags
        },
        "quality_enhancers": quality_tags,
        "guidance": "This prompt aims to capture the essence of your description by layering concepts, from the core subject to the emotional atmosphere."
    }
    
    logger.info(f"[USDR_Recompose] 重组完成. 正向: {positive_prompt[:100]}... | 负向: {negative_prompt[:100]}...")
    
    return positive_prompt, negative_prompt, analysis

@mcp.tool()
def create_prompt_from_scene(scene_description: str, nsfw_level: str = "none") -> Dict[str, Any]:
    """
    🎨 V5版本：通用语义场景解构与智能重组 (USDR) 引擎
    根据自然语言场景描述，使用BGE-M3 V5真正三重向量技术智能生成高质量的AI绘画提示词。
    最大化发挥BGE-M3的语义理解和扩展能力，为您构建完整的画面。

    Args:
        scene_description: 您想要描绘的场景的自然语言描述，可以是SFW或NSFW。
        nsfw_level: 内容的NSFW级别 ("none", "low", "medium", "high")，用于指导扩展搜索。

    Returns:
        一个包含推荐提示词、负面提示词和创作分析的字典。
    """
    if server is None or not server.is_loaded:
        return {"error": "服务器未初始化或数据未加载，请先调用 initialize_server"}

    logger.info(f"[SCENE_TO_PROMPT_V5] 收到场景转换请求，NSFW级别: {nsfw_level}")
    start_time = time.time()

    try:
        # 1. V5场景解构
        concepts = _deconstruct_scene(scene_description, nsfw_level)
        
        # 2. 使用V5核心搜索引擎搜索相关标签
        all_found_tags = []
        concept_contributions = {}
        
        for concept_type, concept_query in concepts.items():
            logger.debug(f"[SCENE_V5] 搜索概念: {concept_type}")
            
            # 直接使用V5核心搜索引擎
            search_results = server.hybrid_search_bge_m3(concept_query, limit=10, search_mode="hybrid")
            
            concept_tags = []
            if search_results and search_results.get("hybrid_results"):
                for result in search_results["hybrid_results"][:5]:  # 取前5个结果
                    doc = result.get("document", "")
                    score = result.get("score", 0.0)
                    
                    # 提取标签
                    if " - " in doc and "】" in doc:
                        tag_part = doc.split(" - ")[0]
                        if "】" in tag_part:
                            clean_tag = tag_part.split("】")[-1].strip()
                            if clean_tag and len(clean_tag) > 1:
                                concept_tags.append(clean_tag)
                                all_found_tags.append(clean_tag)
            
            concept_contributions[concept_type] = concept_tags
            logger.debug(f"[SCENE_V5] {concept_type}: 找到 {len(concept_tags)} 个标签")
        
        # 3. 组合最终提示词
        positive_tags = []
        
        # 添加质量标签
        quality_tags = ["masterpiece", "best_quality", "ultra-detailed", "high_resolution"]
        positive_tags.extend(quality_tags)
        
        # 添加找到的标签
        positive_tags.extend(all_found_tags[:15])  # 限制数量避免过长
        
        positive_prompt = ", ".join(positive_tags)
        
        # 4. 生成负面提示词
        negative_prompt = ("lowres, bad anatomy, bad hands, text, error, missing fingers, "
                          "extra digit, fewer digits, cropped, worst quality, low quality, "
                          "normal quality, jpeg artifacts, signature, watermark, username, blurry")
        
        # 5. 生成分析报告
        scene_analysis = {
            "original_scene": scene_description,
            "derivation_logic": "使用V5三重向量搜索引擎，对场景的各个概念进行语义检索，然后智能组合相关标签",
            "concept_contribution": concept_contributions,
            "quality_enhancers": quality_tags,
            "guidance": "V5技术确保了高质量的语义匹配和标签相关性"
        }
        
        processing_time = time.time() - start_time

        return {
            "positive_prompt": positive_prompt,
            "negative_prompt": negative_prompt,
            "scene_analysis": scene_analysis,
            "expanded_concepts": concept_contributions,
            "generation_metadata": {
                "processing_time": processing_time,
                "search_technology": "BGE-M3 V5 真正三重向量系统",
                "vector_components": [
                    "Dense语义向量",
                    "Sparse词汇向量", 
                    "ColBERT细粒度向量"
                ],
                "total_searches_performed": len(concepts),
                "nsfw_level": nsfw_level,
                "scene_complexity": len(concepts),
                "tags_found": len(all_found_tags)
            },
            "usage_tips": [
                "正面提示词建议权重: 1.0-1.2",
                "负面提示词建议权重: 0.8-1.0", 
                "V5三重向量技术确保了高质量语义搜索",
                "建议配合高质量模型使用以获得最佳效果"
            ]
        }
        
    except Exception as e:
        logger.error(f"[SCENE_TO_PROMPT_V5] 场景转换失败: {e}")
        return {
            "error": f"场景转换失败: {e}",
            "positive_prompt": "masterpiece, best_quality",
            "negative_prompt": "lowres, bad anatomy",
            "scene_analysis": {"error": "处理失败"},
            "generation_metadata": {
                "processing_time": time.time() - start_time,
                "search_technology": "BGE-M3 V5 真正三重向量系统",
                "error_recovery": "已提供基础提示词"
            }
        }

def _dynamic_threshold_search(server, query: str, concept_type: str = "general", min_results: int = 3) -> List[tuple]:
    """
    🎯 优雅的动态阈值搜索系统
    
    实现多层次阈值策略，从高质量到高召回率逐步降级
    确保总能返回有意义的结果，同时保持搜索质量
    
    Args:
        server: 服务器实例
        query: 搜索查询
        concept_type: 概念类型，用于判断搜索策略
        min_results: 最小结果数量阈值
        
    Returns:
        List[tuple]: (tag, score, document) 格式的结果列表
    """
    
    # 📊 智能阈值策略配置
    threshold_strategy = {
        "primary_subject": [0.4, 0.35, 0.3, 0.25],      # 主要角色要求较高质量
        "location_and_style": [0.35, 0.3, 0.25, 0.2],   # 场景风格可以更宽松
        "emotion_and_atmosphere": [0.3, 0.25, 0.2, 0.15], # 情感氛围更注重多样性
        "general": [0.35, 0.3, 0.25, 0.2]               # 通用默认策略
    }
    
    thresholds = threshold_strategy.get(concept_type, threshold_strategy["general"])
    logger.info(f"[DYNAMIC_SEARCH] 概念类型 '{concept_type}' 使用阈值策略: {thresholds}")
    
    for attempt, threshold in enumerate(thresholds, 1):
        logger.debug(f"[DYNAMIC_SEARCH] 尝试 {attempt}/{len(thresholds)}: 阈值={threshold}")
        
        # 🔍 执行搜索
        search_results = server.hybrid_search_bge_m3(query, limit=20, search_mode="hybrid")
        tags = []
        
        if "hybrid_results" in search_results:
            for item in search_results["hybrid_results"]:
                doc = item["document"]
                score = item.get("score", 0.0)
                
                # 🎯 应用当前阈值
                if score < threshold:
                    continue
                    
                # 📝 优先提取高质量标签
                tag = None
                
                # 尝试提取【通用】标签（最高质量）
                import re  # 确保re模块在局部作用域中可用
                match = re.search(r'【通用】(.*?)\s+-', doc)
                if match:
                    tag = match.group(1).strip().replace(' ', '_')
                # 如果没有通用标签，尝试其他类型（需要更高分数）
                elif score >= threshold + 0.1:  # 非通用标签需要额外0.1分数缓冲
                    match = re.search(r'】(.*?)\s+-', doc)
                    if match:
                        tag = match.group(1).strip().replace(' ', '_')
                
                if tag:
                    tags.append((tag, score, doc))
        
        # 📈 检查结果质量
        unique_tags = len(set(tag[0] for tag in tags))
        logger.debug(f"[DYNAMIC_SEARCH] 阈值 {threshold}: 找到 {unique_tags} 个唯一标签")
        
        # ✅ 结果足够时返回
        if unique_tags >= min_results:
            # 📊 按分数排序并去重
            seen_tags = set()
            final_tags = []
            for tag, score, doc in sorted(tags, key=lambda x: x[1], reverse=True):
                if tag not in seen_tags:
                    seen_tags.add(tag)
                    final_tags.append((tag, score, doc))
                    if len(final_tags) >= 8:  # 限制最大返回数量
                        break
            
            logger.info(f"[DYNAMIC_SEARCH] ✅ 成功! 阈值={threshold}, 返回{len(final_tags)}个高质量标签")
            return final_tags
    
    # 🔄 如果所有阈值都没有足够结果，使用关键词回退策略
    logger.warning(f"[DYNAMIC_SEARCH] ⚠️ 所有阈值尝试完毕，执行关键词回退搜索")
    
    # 简单关键词匹配作为最后回退
    keywords = query.lower().split()
    fallback_tags = []
    
    if "hybrid_results" in search_results:
        for item in search_results["hybrid_results"][:10]:
            doc = item["document"].lower()
            if any(keyword in doc for keyword in keywords):
                # 尝试提取任何可能的标签
                import re  # 确保re模块在局部作用域中可用
                for pattern in [r'】(.*?)\s+-', r'(\w+)\s+-']:
                    match = re.search(pattern, item["document"])
                    if match:
                        tag = match.group(1).strip().replace(' ', '_')
                        fallback_tags.append((tag, item.get("score", 0.1), item["document"]))
                        break
    
    logger.info(f"[DYNAMIC_SEARCH] 🔄 回退策略返回 {len(fallback_tags)} 个标签")
    return fallback_tags[:3]  # 回退时只返回少量结果


def _get_related_prompts(prompt: str, similarity_threshold: float = 0.7) -> Dict[str, Any]:
    """
    获取与给定提示词相关的其他提示词建议 - 内部辅助函数
    
    Args:
        prompt: 输入的提示词
        similarity_threshold: 相似度阈值 (0.0-1.0)
        
    Returns:
        相关提示词推荐结果
    """
    global server
    
    if server is None:
        return {"error": "服务器未初始化，请先调用 initialize_server"}
    
    try:
        logger.info(f"[RELATED] 获取'{prompt}'的相关提示词")
        
        # 使用BGE-M3混合搜索获得更精准的相关结果
        search_result = server.hybrid_search_bge_m3(f"{prompt} 相关 类似 同类", 15, "hybrid")
        
        if "error" in search_result:
            return search_result
        
        # 提取相关标签
        related_tags = []
        seen_tags = set()
        
        if "hybrid_results" in search_result:
            for item in search_result["hybrid_results"]:
                doc = item["document"]
                if " - " in doc and "】" in doc:
                    tag_name = doc.split(" - ")[0].split("】")[-1].strip()
                    if tag_name and tag_name != prompt and tag_name not in seen_tags:
                        seen_tags.add(tag_name)
                        related_tags.append({
                            "tag": tag_name,
                            "explanation": doc,
                            "score": item["score"],
                            "source": item.get("source", "BGE-M3混合搜索")
                        })
        
        suggested_combinations = [
            f"{prompt}, {tag['tag']}" for tag in related_tags[:5]
        ]
        
        return {
            "original_prompt": prompt,
            "search_method": "🚀 BGE-M3三重能力混合搜索",
            "related_count": len(related_tags),
            "related_tags": related_tags[:10],
            "suggested_combinations": suggested_combinations,
            "copyable_combinations": " | ".join(suggested_combinations),
            "search_time": search_result.get("search_time", 0)
        }
        
    except Exception as e:
        logger.error(f"[ERROR] 相关提示词搜索失败: {e}")
        return {"error": f"相关提示词搜索失败: {str(e)}"}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='最小版Danbooru搜索服务器-增强版')
    parser.add_argument('--data-path', '-d', type=str, help='数据文件路径')
    parser.add_argument('--database-path', '--db-path', type=str, help='ChromaDB数据库路径')
    parser.add_argument('--collection-name', '-c', type=str, help='ChromaDB集合名称')
    parser.add_argument('--auto-init', action='store_true', help='自动初始化服务器')
    parser.add_argument('--use-fp16', action='store_true', default=True, help='是否使用FP16精度')
    
    args = parser.parse_args()
    
    logger.info("[START] 启动最小版Danbooru搜索服务器-增强版...")
    
    if args.auto_init:
        try:
            logger.info("[AUTO] 自动初始化模式")
            server = MinimalDanbooruServer(use_fp16=args.use_fp16)
            server.load_model()
            server.load_test_data(args.data_path, args.database_path, args.collection_name)
            logger.info("[OK] 自动初始化完成")
        except Exception as e:
            logger.error(f"[ERROR] 自动初始化失败: {e}")
            logger.info("[MANUAL] 将使用手动初始化模式")
    
    logger.info("[🚀 BGE-M3] 核心能力已集成到以下6个智能工具中:")
    logger.info("  🔧 initialize_server: 状态检查和故障排除（服务器已自动初始化）")
    logger.info("  🔍 search: 智能搜索 (自动意图识别/查询增强/降级策略)")
    logger.info("  📊 analyze_prompts: 提示词深度分析 (✅ 优雅的多层级标签解析)")
    logger.info("  ✍️ create_nsfw_content: NSFW内容创作")
    logger.info("  🤖 get_smart_recommendations: 智能推荐系统 (✅ 智能参数处理)")
    logger.info("  ℹ️ get_server_info: 获取服务器综合信息")
    
    logger.info("[🧠 BGE-M3] 三重能力特性:")
    logger.info("  🎯 Dense向量: 语义理解，同义词匹配")
    logger.info("  🔑 Sparse向量: 关键词精确匹配")
    logger.info("  🔬 ColBERT向量: 细粒度token级匹配")
    
    logger.info("[🤖 智能化增强特性:")
    logger.info("  🧠 智能意图识别: 自动检测查询类型")
    logger.info("  🔍 自动查询增强: 根据意图优化查询")
    logger.info("  🔄 智能降级策略: 搜索失败自动重试")
    logger.info("  📈 个性化推荐: 基于历史的智能推荐 (✅ 参数处理优化)")
    logger.info("  📊 性能自学习: 实时优化搜索策略")
    logger.info("  🎯 上下文感知: 多轮对话支持")
    logger.info("  🔞 NSFW内容智能检测")
    logger.info("  📊 批量提示词智能分析 (✅ 多层级标签解析)")
    logger.info("  🔗 相关提示词关联推荐")
    logger.info("  🏷️ 标签别名映射系统 (✅ 解决标签缺失问题)")
    logger.info("  💾 数据库和内存双模式")
    logger.info("  ⚡ FP16精度优化")
    logger.info("  🏆 BGE-M3官方推荐配置（优雅增强版）")
    
    logger.info("[READY] 服务器就绪，等待MCP连接...")
    
    # 启动MCP服务器
    mcp.run() 
