#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🎨 Danbooru BGE-M3 RAG Server - 基础使用示例
===============================================

本文件展示了如何使用Danbooru BGE-M3 RAG服务器的基本功能。
"""

import asyncio
import json
from typing import Dict, List, Any

# 模拟MCP客户端连接
class DanbooruRAGClient:
    """Danbooru RAG客户端示例"""
    
    def __init__(self, server_url: str = "http://localhost:8000"):
        self.server_url = server_url
        print(f"🔗 连接到服务器: {server_url}")
    
    async def search(self, query: str, limit: int = 10) -> Dict[str, Any]:
        """基础搜索功能"""
        print(f"🔍 搜索查询: '{query}'")
        # 这里模拟API调用
        return {
            "query": query,
            "results": [
                {"tag": "1girl", "translation": "单个女性", "confidence": 0.95},
                {"tag": "blonde_hair", "translation": "金发", "confidence": 0.92},
                {"tag": "blue_eyes", "translation": "蓝眼睛", "confidence": 0.90}
            ],
            "total_found": 156,
            "search_time": "0.15s"
        }

async def example_basic_search():
    """示例1: 基础标签搜索"""
    print("\n" + "="*50)
    print("📝 示例1: 基础标签搜索")
    print("="*50)
    
    client = DanbooruRAGClient()
    
    # 搜索基础标签
    queries = [
        "1girl blonde hair blue eyes",
        "school uniform anime style",
        "cat ears cute girl",
        "sunset beach scenery"
    ]
    
    for query in queries:
        result = await client.search(query)
        print(f"\n✨ 查询: {query}")
        print(f"📊 找到 {result['total_found']} 个相关标签")
        print(f"⏱️ 搜索耗时: {result['search_time']}")
        
        print("🏷️ 推荐标签:")
        for i, tag in enumerate(result['results'][:3], 1):
            print(f"  {i}. {tag['tag']} - {tag['translation']} (置信度: {tag['confidence']:.2f})")

async def example_artist_search():
    """示例2: 艺术家搜索"""
    print("\n" + "="*50)
    print("🎨 示例2: 艺术家搜索")
    print("="*50)
    
    client = DanbooruRAGClient()
    
    # 艺术家搜索示例
    artist_queries = [
        "artist:kantoku",
        "画师 缺少一半的蓝色",
        "artist:khyle",
        "realistic style artist"
    ]
    
    for query in artist_queries:
        result = await client.search(query)
        print(f"\n🎭 查询: {query}")
        if "artist:" in query.lower():
            print("💡 检测到艺术家搜索模式")
        print(f"📈 相关度评分: 92%")
        print("🎨 推荐艺术家和风格:")
        
        artists = [
            {"name": "kantoku", "style": "软萌校园风", "works": "4,521"},
            {"name": "khyle", "style": "碧蓝档案专家", "works": "2,156"},
            {"name": "缺少一半的蓝色", "style": "清新插画风", "works": "1,823"}
        ]
        
        for artist in artists:
            print(f"  ✨ {artist['name']} - {artist['style']} ({artist['works']}作品)")

async def example_scene_generation():
    """示例3: 场景描述生成"""
    print("\n" + "="*50)
    print("🌅 示例3: 场景描述生成")
    print("="*50)
    
    client = DanbooruRAGClient()
    
    # 场景描述示例
    scenes = [
        "一个女孩在海边看日落",
        "学校图书馆里的安静下午",
        "樱花飞舞的春日校园",
        "咖啡厅里的温馨时光"
    ]
    
    for scene in scenes:
        print(f"\n🎬 场景描述: '{scene}'")
        print("🔄 正在分析场景要素...")
        
        # 模拟场景分析结果
        elements = {
            "characters": ["1girl", "solo"],
            "setting": ["beach", "sunset", "ocean"],
            "mood": ["peaceful", "romantic", "warm"],
            "style": ["anime", "soft_lighting"]
        }
        
        print("🧩 场景要素分解:")
        for category, tags in elements.items():
            print(f"  📋 {category}: {', '.join(tags)}")
        
        # 生成完整提示词
        full_prompt = "1girl, solo, beach, sunset, ocean, peaceful, romantic, warm lighting, anime style, soft colors, beautiful detailed"
        print(f"\n📝 生成的完整提示词:")
        print(f"  {full_prompt}")

async def example_multilingual_search():
    """示例4: 多语言搜索"""
    print("\n" + "="*50)
    print("🌍 示例4: 多语言搜索")
    print("="*50)
    
    client = DanbooruRAGClient()
    
    # 多语言查询示例
    multilingual_queries = [
        ("中文", "可爱的猫女孩"),
        ("英文", "cute cat girl"),
        ("日文", "かわいい猫耳"),
        ("混合", "1girl cute 猫耳 kawaii")
    ]
    
    for lang, query in multilingual_queries:
        print(f"\n🗣️ {lang}查询: '{query}'")
        result = await client.search(query)
        
        print("🔤 语言检测和标准化:")
        if lang == "中文":
            print("  检测: 中文 → 转换为Danbooru标签")
            print("  结果: cat_ears, 1girl, cute, animal_ears")
        elif lang == "日文":
            print("  检测: 日文 → 转换为Danbooru标签")
            print("  结果: cat_ears, kawaii, moe")
        else:
            print("  检测: 英文 → 直接匹配")
        
        print(f"📊 找到 {result['total_found']} 个匹配标签")

async def example_nsfw_content():
    """示例5: NSFW内容生成 (仅用于演示)"""
    print("\n" + "="*50)
    print("🔞 示例5: NSFW内容生成")
    print("="*50)
    
    print("⚠️ 注意: 这是NSFW内容生成的技术演示")
    print("🛡️ 实际使用需要适当的安全过滤和用户同意")
    
    # NSFW提示词示例 (教育目的)
    nsfw_prompts = [
        "1girl, swimsuit, beach",
        "bath scene, steam, towel",
        "bedroom, morning light"
    ]
    
    for prompt in nsfw_prompts:
        print(f"\n📝 输入提示词: '{prompt}'")
        print("🔍 NSFW级别检测: 低风险")
        print("🎨 风格化建议:")
        print("  - 艺术化表现")
        print("  - 适度遮挡")
        print("  - 唯美构图")
        
        enhanced_prompt = f"{prompt}, artistic, beautiful, aesthetic, masterpiece"
        print(f"✨ 增强后提示词: {enhanced_prompt}")

async def example_batch_processing():
    """示例6: 批量处理"""
    print("\n" + "="*50)
    print("📦 示例6: 批量处理")
    print("="*50)
    
    client = DanbooruRAGClient()
    
    # 批量处理示例
    batch_queries = [
        "school girl uniform",
        "magical girl transformation",
        "maid cafe waitress",
        "library study scene",
        "garden tea party"
    ]
    
    print(f"🔄 开始批量处理 {len(batch_queries)} 个查询...")
    
    results = []
    for i, query in enumerate(batch_queries, 1):
        print(f"📊 处理进度: {i}/{len(batch_queries)} - {query}")
        result = await client.search(query)
        results.append(result)
        
        # 模拟处理时间
        await asyncio.sleep(0.1)
    
    print("\n✅ 批量处理完成!")
    print(f"📈 总计处理: {len(results)} 个查询")
    print(f"⏱️ 平均响应时间: 0.15s")
    print(f"🎯 成功率: 100%")
    
    # 显示统计信息
    total_tags = sum(r['total_found'] for r in results)
    print(f"🏷️ 总计找到标签: {total_tags:,}")

async def main():
    """主函数 - 运行所有示例"""
    print("🎨 Danbooru BGE-M3 RAG Server - 使用示例集合")
    print("=" * 60)
    print("📚 本示例展示了服务器的各种功能用法")
    print("💡 适合初学者了解和学习系统功能")
    print("=" * 60)
    
    # 运行所有示例
    await example_basic_search()
    await example_artist_search()
    await example_scene_generation()
    await example_multilingual_search()
    await example_nsfw_content()
    await example_batch_processing()
    
    print("\n" + "="*60)
    print("🎉 所有示例运行完成!")
    print("📖 更多高级功能请查看 advanced_examples.py")
    print("🔗 API文档: https://github.com/2799662352/rag-mcp")
    print("="*60)

if __name__ == "__main__":
    # 运行示例
    asyncio.run(main())