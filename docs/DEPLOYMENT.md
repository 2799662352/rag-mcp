# 🚀 部署指南

## 快速部署

### 1. 环境要求
```bash
# Python 3.11+
# CUDA 11.8+ (可选，用于GPU加速)
# 16GB+ RAM
# 50GB+ 磁盘空间
```

### 2. 安装依赖
```bash
git clone https://github.com/2799662352/rag-mcp.git
cd rag-mcp
pip install -r requirements.txt
```

### 3. 启动服务器
```bash
python danbooru_prompt_server_v2_minimal.py
```

## Docker部署

### 使用Docker Compose
```bash
docker-compose up -d
```

### 单独Docker运行
```bash
docker build -t danbooru-rag .
docker run -p 8000:8000 danbooru-rag
```

## 生产环境部署

### 使用uWSGI
```bash
pip install uwsgi
uwsgi --http :8000 --module danbooru_prompt_server_v2_minimal:app
```

### 使用Gunicorn
```bash
pip install gunicorn
gunicorn -w 4 -b 0.0.0.0:8000 danbooru_prompt_server_v2_minimal:app
```

## 配置说明

详细配置请参考 `config.yaml` 文件。

## 性能优化

- 启用GPU加速可提升5-10x性能
- 配置Redis缓存可减少重复计算
- 使用SSD存储可提升I/O性能

## 监控和日志

系统提供完整的监控和日志功能，详情请查看 `logs/` 目录。