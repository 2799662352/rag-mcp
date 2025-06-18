#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🎯 Danbooru数据预处理脚本
========================

将Danbooru标签数据转换为训练用的JSONL格式，支持多种数据源：
- 本地Danbooru数据集
- 在线Danbooru API
- 自定义JSON格式

输出格式:
{
    "id": "unique_id",
    "text": "标签描述文本",
    "source": "danbooru",
    "metadata": {
        "category": "artist/character/copyright/general/meta",
        "count": 使用次数,
        "rating": 评级
    }
}
"""

import json
import argparse
import requests
from typing import Dict, List, Any, Optional
import time
from pathlib import Path
import pandas as pd
from tqdm import tqdm
import logging

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DanbooruDataPreparer:
    """Danbooru数据预处理器"""
    
    def __init__(self, output_file: str, max_items: Optional[int] = None):
        """
        初始化数据预处理器
        
        Args:
            output_file: 输出JSONL文件路径
            max_items: 最大处理条目数（用于测试）
        """
        self.output_file = output_file
        self.max_items = max_items
        self.processed_count = 0
        
        # 确保输出目录存在
        Path(output_file).parent.mkdir(parents=True, exist_ok=True)
        
        # 标签类别映射
        self.category_map = {
            0: "general",
            1: "artist", 
            3: "copyright",
            4: "character",
            5: "meta"
        }
        
        # 标签描述模板
        self.tag_templates = {
            "artist": "【画师】{tag} - 专门创作{style_desc}风格作品的艺术家",
            "character": "【角色】{tag} - 来自{series}的角色，{char_desc}",
            "copyright": "【版权】{tag} - {series_desc}",
            "general": "【通用】{tag} - {general_desc}",
            "meta": "【元数据】{tag} - {meta_desc}"
        }
    
    def prepare_from_local_dataset(self, dataset_path: str, format_type: str = "parquet") -> None:
        """
        从本地数据集准备数据
        
        Args:
            dataset_path: 数据集文件路径
            format_type: 数据格式 ("parquet", "csv", "json")
        """
        logger.info(f"Loading local dataset from {dataset_path}")
        
        try:
            if format_type == "parquet":
                df = pd.read_parquet(dataset_path)
            elif format_type == "csv":
                df = pd.read_csv(dataset_path)
            elif format_type == "json":
                df = pd.read_json(dataset_path)
            else:
                raise ValueError(f"Unsupported format: {format_type}")
                
            logger.info(f"Loaded {len(df)} records")
            
            # 如果有最大条目限制
            if self.max_items:
                df = df.head(self.max_items)
                logger.info(f"Limited to {len(df)} records")
            
            self._process_dataframe(df)
            
        except Exception as e:
            logger.error(f"Error loading dataset: {e}")
            raise
    
    def prepare_from_danbooru_api(self, 
                                 api_key: Optional[str] = None,
                                 limit: int = 1000,
                                 tags_only: bool = True) -> None:
        """
        从Danbooru API准备数据
        
        Args:
            api_key: Danbooru API密钥
            limit: 获取条目数限制
            tags_only: 是否只获取标签数据
        """
        logger.info("Fetching data from Danbooru API")
        
        base_url = "https://danbooru.donmai.us"
        headers = {}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        
        try:
            if tags_only:
                # 获取标签数据
                url = f"{base_url}/tags.json"
                params = {"limit": min(limit, 1000), "search[order]": "count"}
                
                response = requests.get(url, headers=headers, params=params)
                response.raise_for_status()
                
                tags_data = response.json()
                self._process_tags_data(tags_data)
            else:
                # 获取帖子数据
                url = f"{base_url}/posts.json"
                params = {"limit": min(limit, 200), "tags": "rating:s"}
                
                response = requests.get(url, headers=headers, params=params)
                response.raise_for_status()
                
                posts_data = response.json()
                self._process_posts_data(posts_data)
                
        except requests.RequestException as e:
            logger.error(f"API request failed: {e}")
            raise
    
    def prepare_from_tag_list(self, tags_file: str) -> None:
        """
        从标签列表文件准备数据
        
        Args:
            tags_file: 标签列表文件路径
        """
        logger.info(f"Processing tags from {tags_file}")
        
        try:
            with open(tags_file, 'r', encoding='utf-8') as f:
                if tags_file.endswith('.json'):
                    tags_data = json.load(f)
                    if isinstance(tags_data, dict):
                        # 处理分类标签格式
                        self._process_categorized_tags(tags_data)
                    else:
                        # 处理标签列表格式
                        self._process_tag_list(tags_data)
                else:
                    # 处理纯文本标签文件
                    tags = [line.strip() for line in f if line.strip()]
                    self._process_tag_list(tags)
                    
        except Exception as e:
            logger.error(f"Error processing tags file: {e}")
            raise
    
    def _process_dataframe(self, df: pd.DataFrame) -> None:
        """处理DataFrame格式的数据"""
        with open(self.output_file, 'w', encoding='utf-8') as f:
            for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing records"):
                try:
                    # 提取标签
                    if 'tag_string' in row:
                        tags = row['tag_string'].split()
                    elif 'tags' in row:
                        tags = row['tags'] if isinstance(row['tags'], list) else row['tags'].split()
                    else:
                        continue
                    
                    # 创建训练数据条目
                    for tag in tags:
                        if not tag.strip():
                            continue
                            
                        entry = {
                            "id": f"post_{row.get('id', idx)}_{tag}",
                            "text": self._create_tag_description(tag, category="general"),
                            "source": "danbooru",
                            "metadata": {
                                "tag": tag,
                                "post_id": row.get('id', idx),
                                "category": "general",
                                "rating": row.get('rating', 's')
                            }
                        }
                        
                        f.write(json.dumps(entry, ensure_ascii=False) + '\n')
                        self.processed_count += 1
                        
                        if self.max_items and self.processed_count >= self.max_items:
                            break
                            
                except Exception as e:
                    logger.warning(f"Error processing row {idx}: {e}")
                    continue
                    
                if self.max_items and self.processed_count >= self.max_items:
                    break
    
    def _process_tags_data(self, tags_data: List[Dict]) -> None:
        """处理API返回的标签数据"""
        with open(self.output_file, 'w', encoding='utf-8') as f:
            for tag_info in tqdm(tags_data, desc="Processing tags"):
                try:
                    tag_name = tag_info.get('name', '')
                    category_id = tag_info.get('category', 0)
                    category = self.category_map.get(category_id, 'general')
                    count = tag_info.get('post_count', 0)
                    
                    entry = {
                        "id": f"tag_{tag_info.get('id', tag_name)}",
                        "text": self._create_tag_description(tag_name, category, count),
                        "source": "danbooru",
                        "metadata": {
                            "tag": tag_name,
                            "category": category,
                            "count": count,
                            "tag_id": tag_info.get('id')
                        }
                    }
                    
                    f.write(json.dumps(entry, ensure_ascii=False) + '\n')
                    self.processed_count += 1
                    
                    if self.max_items and self.processed_count >= self.max_items:
                        break
                        
                except Exception as e:
                    logger.warning(f"Error processing tag: {e}")
                    continue
    
    def _process_posts_data(self, posts_data: List[Dict]) -> None:
        """处理API返回的帖子数据"""
        with open(self.output_file, 'w', encoding='utf-8') as f:
            for post in tqdm(posts_data, desc="Processing posts"):
                try:
                    # 合并所有标签
                    all_tags = []
                    for cat in ['tag_string_general', 'tag_string_artist', 
                              'tag_string_character', 'tag_string_copyright']:
                        tags = post.get(cat, '').split()
                        all_tags.extend(tags)
                    
                    # 创建帖子描述
                    post_desc = f"包含标签: {', '.join(all_tags[:10])}"
                    if len(all_tags) > 10:
                        post_desc += f" 等共{len(all_tags)}个标签"
                    
                    entry = {
                        "id": f"post_{post['id']}",
                        "text": post_desc,
                        "source": "danbooru",
                        "metadata": {
                            "post_id": post['id'],
                            "rating": post.get('rating', 's'),
                            "score": post.get('score', 0),
                            "tags_count": len(all_tags)
                        }
                    }
                    
                    f.write(json.dumps(entry, ensure_ascii=False) + '\n')
                    self.processed_count += 1
                    
                    if self.max_items and self.processed_count >= self.max_items:
                        break
                        
                except Exception as e:
                    logger.warning(f"Error processing post: {e}")
                    continue
    
    def _process_categorized_tags(self, tags_data: Dict) -> None:
        """处理分类标签数据"""
        with open(self.output_file, 'w', encoding='utf-8') as f:
            for category, tags in tags_data.items():
                logger.info(f"Processing {len(tags)} {category} tags")
                
                for tag in tqdm(tags, desc=f"Processing {category}"):
                    try:
                        if isinstance(tag, dict):
                            tag_name = tag.get('name', '')
                            count = tag.get('count', 0)
                        else:
                            tag_name = str(tag)
                            count = 0
                        
                        if not tag_name.strip():
                            continue
                        
                        entry = {
                            "id": f"{category}_{tag_name}",
                            "text": self._create_tag_description(tag_name, category, count),
                            "source": "danbooru",
                            "metadata": {
                                "tag": tag_name,
                                "category": category,
                                "count": count
                            }
                        }
                        
                        f.write(json.dumps(entry, ensure_ascii=False) + '\n')
                        self.processed_count += 1
                        
                        if self.max_items and self.processed_count >= self.max_items:
                            return
                            
                    except Exception as e:
                        logger.warning(f"Error processing tag {tag}: {e}")
                        continue
    
    def _process_tag_list(self, tags: List[str]) -> None:
        """处理标签列表"""
        with open(self.output_file, 'w', encoding='utf-8') as f:
            for i, tag in enumerate(tqdm(tags, desc="Processing tags")):
                try:
                    if not tag.strip():
                        continue
                    
                    entry = {
                        "id": f"tag_{i}_{tag}",
                        "text": self._create_tag_description(tag),
                        "source": "danbooru",
                        "metadata": {
                            "tag": tag,
                            "category": "general",
                            "index": i
                        }
                    }
                    
                    f.write(json.dumps(entry, ensure_ascii=False) + '\n')
                    self.processed_count += 1
                    
                    if self.max_items and self.processed_count >= self.max_items:
                        break
                        
                except Exception as e:
                    logger.warning(f"Error processing tag {tag}: {e}")
                    continue
    
    def _create_tag_description(self, tag: str, category: str = "general", count: int = 0) -> str:
        """创建标签描述文本"""
        # 基础描述
        if category == "artist":
            desc = f"【画师】{tag} - AI绘画提示词中的艺术家标签，用于指定特定画师的艺术风格"
        elif category == "character":
            desc = f"【角色】{tag} - AI绘画提示词中的角色标签，用于生成特定动漫/游戏角色"
        elif category == "copyright":
            desc = f"【版权】{tag} - AI绘画提示词中的作品标签，表示特定的动漫、游戏或其他媒体作品"
        elif category == "meta":
            desc = f"【元数据】{tag} - AI绘画提示词中的技术标签，用于控制图像的技术特征"
        else:  # general
            desc = f"【通用】{tag} - AI绘画提示词中的描述标签，用于描述图像内容、风格或特征"
        
        # 添加使用频率信息
        if count > 0:
            if count > 100000:
                freq_desc = "（非常常用）"
            elif count > 10000:
                freq_desc = "（常用）"
            elif count > 1000:
                freq_desc = "（较常用）"
            else:
                freq_desc = "（较少用）"
            desc += freq_desc
        
        return desc
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取处理统计信息"""
        return {
            "processed_count": self.processed_count,
            "output_file": self.output_file,
            "file_size": Path(self.output_file).stat().st_size if Path(self.output_file).exists() else 0
        }


def main():
    parser = argparse.ArgumentParser(description="Prepare Danbooru data for training")
    parser.add_argument("--output", "-o", default="danbooru_tags.jsonl", 
                       help="Output JSONL file path")
    parser.add_argument("--source-type", choices=["local", "api", "tags"], 
                       default="local", help="Data source type")
    parser.add_argument("--source-path", help="Path to local dataset or tags file")
    parser.add_argument("--format", choices=["parquet", "csv", "json", "txt"], 
                       default="parquet", help="Input data format")
    parser.add_argument("--api-key", help="Danbooru API key")
    parser.add_argument("--limit", type=int, default=1000, 
                       help="Maximum number of items to process")
    parser.add_argument("--max-items", type=int, 
                       help="Maximum items for testing")
    
    args = parser.parse_args()
    
    preparer = DanbooruDataPreparer(args.output, args.max_items)
    
    try:
        if args.source_type == "local":
            if not args.source_path:
                raise ValueError("--source-path is required for local data source")
            preparer.prepare_from_local_dataset(args.source_path, args.format)
            
        elif args.source_type == "api":
            preparer.prepare_from_danbooru_api(args.api_key, args.limit)
            
        elif args.source_type == "tags":
            if not args.source_path:
                raise ValueError("--source-path is required for tags data source")
            preparer.prepare_from_tag_list(args.source_path)
        
        # 输出统计信息
        stats = preparer.get_statistics()
        logger.info("Processing completed!")
        logger.info(f"Processed items: {stats['processed_count']}")
        logger.info(f"Output file: {stats['output_file']}")
        logger.info(f"File size: {stats['file_size']} bytes")
        
    except Exception as e:
        logger.error(f"Processing failed: {e}")
        raise


if __name__ == "__main__":
    main()