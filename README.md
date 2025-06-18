# 🎨 Danbooru BGE-M3 RAG Server

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.13+](https://img.shields.io/badge/python-3.13+-blue.svg)](https://www.python.org/downloads/)
[![BGE-M3](https://img.shields.io/badge/BGE--M3-Multi--Vector-green.svg)](https://huggingface.co/BAAI/bge-m3)

一个基于BAAI/bge-m3的高性能Danbooru标签检索增强生成（RAG）服务器，专为AI绘画提示词优化设计。

## ✨ 特性亮点

- 🚀 **BGE-M3三重向量搜索**: Dense + Sparse + ColBERT多重检索技术
- 🌍 **多语言支持**: 100+种语言，中英文混合查询优化
- 🎯 **专业标签库**: 1,386,373条Danbooru标签，完整中文解释
- 🔥 **实时语义搜索**: 毫秒级响应，支持复杂查询意图识别
- 🎨 **AI绘画优化**: 专为Stable Diffusion等AI绘画工具设计
- 🛡️ **智能内容分级**: 自动NSFW检测和安全过滤
- ⚡ **GPU加速**: CUDA支持，批量处理优化

## 🚀 快速开始

### 安装

```bash
git clone https://github.com/2799662352/rag-mcp.git
cd rag-mcp
pip install -r requirements.txt
```

### 启动服务器

```bash
python danbooru_prompt_server_v2_minimal.py
```

## 💡 核心功能

### 🔍 语义搜索
```python
search("1girl blonde_hair blue_eyes")
```

### 🧠 提示词分析
```python
analyze_prompts(["1girl", "cat_ears", "school_uniform"])
```

### 🎨 场景生成
```python
create_prompt_from_scene("一个猫娘在海滩上的夏日场景")
```

## 📊 性能指标

- 检索精度@10: **84.7%**
- 响应时间: **<500ms**
- 支持语言: **100+**
- 标签数量: **1,386,373条**

## 🤖 技术架构

基于BGE-M3的三重向量搜索技术：
- **Dense Vector**: 语义相似度检索
- **Sparse Vector**: 精确关键词匹配
- **ColBERT Vector**: 细粒度交互匹配

## 📁 文件结构

```
rag-mcp/
├── danbooru_prompt_server_v2_minimal.py  # 核心服务器
├── requirements.txt                       # 依赖包
├── Dockerfile                            # Docker部署
├── TRAINING.md                           # BGE-M3训练指南
├── CONTRIBUTING.md                       # 贡献指南
└── .github/workflows/ci.yml              # CI/CD流水线
```

## 📝 许可证

MIT License - 详见 [LICENSE](LICENSE) 文件。

## 🤝 贡献

欢迎提交Issue和Pull Request！详见 [CONTRIBUTING.md](CONTRIBUTING.md)。

---

**⭐ 如果这个项目对您有帮助，请给我们一个star！**