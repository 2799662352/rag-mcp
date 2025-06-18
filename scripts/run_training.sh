#!/bin/bash
# 🚀 BGE-M3 Danbooru训练一键脚本
# =================================

set -e  # 遇到错误立即退出

echo "🎯 Starting BGE-M3 Danbooru Training Pipeline..."

# 配置参数
MODEL_NAME="BAAI/bge-m3"
DB_DIR="artifacts/vector_stores/chroma_db"
OUTPUT_FILE="danbooru_tags.jsonl"
BATCH_SIZE=32
MAX_ITEMS=""

# 解析命令行参数
while [[ $# -gt 0 ]]; do
    case $1 in
        --model)
            MODEL_NAME="$2"
            shift 2
            ;;
        --batch-size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --max-items)
            MAX_ITEMS="--max-items $2"
            shift 2
            ;;
        --db-dir)
            DB_DIR="$2"
            shift 2
            ;;
        --help)
            echo "Usage: $0 [options]"
            echo "Options:"
            echo "  --model MODEL_NAME    BGE model to use (default: BAAI/bge-m3)"
            echo "  --batch-size SIZE     Batch size for training (default: 32)"
            echo "  --max-items NUM       Maximum items to process (for testing)"
            echo "  --db-dir DIR          ChromaDB directory (default: artifacts/vector_stores/chroma_db)"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# 检查Python环境
echo "📋 Checking Python environment..."
python3 --version
if ! python3 -c "import torch; print(f'PyTorch: {torch.__version__}')"; then
    echo "❌ PyTorch not found. Please install it first."
    exit 1
fi

if ! python3 -c "import FlagEmbedding; print('FlagEmbedding: OK')"; then
    echo "❌ FlagEmbedding not found. Installing..."
    pip install FlagEmbedding>=1.3.0
fi

# 检查GPU
echo "🔍 Checking GPU availability..."
if python3 -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"; then
    python3 -c "import torch; print(f'GPU name: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"
    python3 -c "import torch; print(f'GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB' if torch.cuda.is_available() else 'N/A')"
fi

# 创建必要目录
echo "📁 Creating directories..."
mkdir -p artifacts/vector_stores
mkdir -p artifacts/logs
mkdir -p artifacts/models

# 步骤1: 数据准备
echo "🔄 Step 1: Preparing Danbooru data..."
if [[ ! -f "$OUTPUT_FILE" ]]; then
    echo "📥 Preparing data from local sources..."
    
    # 检查是否有本地数据集
    if [[ -f "danbooru2024_complete.parquet" ]]; then
        python3 prepare_danbooru_data.py \
            --source-type local \
            --source-path danbooru2024_complete.parquet \
            --format parquet \
            --output "$OUTPUT_FILE" \
            $MAX_ITEMS
    elif [[ -f "danbooru_tags_2024.txt" ]]; then
        python3 prepare_danbooru_data.py \
            --source-type tags \
            --source-path danbooru_tags_2024.txt \
            --output "$OUTPUT_FILE" \
            $MAX_ITEMS
    else
        echo "⚠️  No local dataset found. Creating sample data..."
        python3 -c "
import json
sample_data = [
    {'id': '1', 'text': '【通用】1girl - 单个女性角色。这是最基础的角色标签', 'source': 'danbooru'},
    {'id': '2', 'text': '【服装】school_uniform - 学生制服。包括水手服、西式制服等', 'source': 'danbooru'},
    {'id': '3', 'text': '【画师】kantoku - 专门创作萌系风格作品的著名插画师', 'source': 'danbooru'},
    {'id': '4', 'text': '【角色】hatsune_miku - 来自VOCALOID的虚拟歌手，双马尾绿发特征', 'source': 'danbooru'},
    {'id': '5', 'text': '【通用】masterpiece - 高质量作品标签，用于提升图像整体质量', 'source': 'danbooru'}
]
with open('$OUTPUT_FILE', 'w', encoding='utf-8') as f:
    for item in sample_data:
        f.write(json.dumps(item, ensure_ascii=False) + '\n')
print(f'Created sample data: $OUTPUT_FILE')
"
    fi
else
    echo "✅ Data file already exists: $OUTPUT_FILE"
fi

# 验证数据文件
echo "🔍 Validating data file..."
if [[ -f "$OUTPUT_FILE" ]]; then
    line_count=$(wc -l < "$OUTPUT_FILE")
    echo "📊 Data file contains $line_count lines"
    echo "📝 Sample data:"
    head -3 "$OUTPUT_FILE"
else
    echo "❌ Data file not found: $OUTPUT_FILE"
    exit 1
fi

# 步骤2: 向量化训练
echo "🚀 Step 2: Starting vectorization training..."
echo "🎯 Model: $MODEL_NAME"
echo "📦 Batch size: $BATCH_SIZE"
echo "💾 Database: $DB_DIR"

# 设置环境变量
export TOKENIZERS_PARALLELISM=false
export TRANSFORMERS_VERBOSITY=error

# 根据GPU内存调整批次大小
if python3 -c "import torch; exit(0 if torch.cuda.is_available() and torch.cuda.get_device_properties(0).total_memory < 8e9 else 1)"; then
    echo "⚠️  Low GPU memory detected. Reducing batch size to 16"
    BATCH_SIZE=16
fi

# 执行训练
python3 vectorizer.py \
    --input "$OUTPUT_FILE" \
    --db "$DB_DIR" \
    --model "$MODEL_NAME" \
    --batch-size "$BATCH_SIZE" \
    2>&1 | tee "artifacts/logs/training_$(date +%Y%m%d_%H%M%S).log"

# 检查训练结果
echo "🔍 Checking training results..."
if [[ -d "$DB_DIR" ]]; then
    echo "✅ Vector database created successfully"
    
    # 显示数据库统计
    python3 -c "
import chromadb
try:
    client = chromadb.PersistentClient(path='$DB_DIR')
    collections = client.list_collections()
    for collection in collections:
        count = collection.count()
        print(f'Collection: {collection.name}, Documents: {count}')
except Exception as e:
    print(f'Error reading database: {e}')
"
else
    echo "❌ Vector database not found"
    exit 1
fi

# 步骤3: 测试搜索功能
echo "🧪 Step 3: Testing search functionality..."
python3 -c "
import chromadb
import sys

try:
    client = chromadb.PersistentClient(path='$DB_DIR')
    collections = client.list_collections()
    
    if not collections:
        print('❌ No collections found')
        sys.exit(1)
    
    collection = collections[0]
    print(f'🔍 Testing search on collection: {collection.name}')
    
    # 测试查询
    test_queries = ['1girl anime', 'school uniform', 'high quality']
    
    for query in test_queries:
        try:
            results = collection.query(
                query_texts=[query],
                n_results=3
            )
            print(f'Query: \"{query}\" -> {len(results[\"documents\"][0])} results')
            for i, doc in enumerate(results['documents'][0][:2]):
                print(f'  {i+1}. {doc[:100]}...')
        except Exception as e:
            print(f'Query error for \"{query}\": {e}')
    
    print('✅ Search functionality test completed')
    
except Exception as e:
    print(f'❌ Search test failed: {e}')
    sys.exit(1)
"

# 完成
echo "🎉 Training pipeline completed successfully!"
echo "📍 Vector database location: $DB_DIR"
echo "📄 Training logs: artifacts/logs/"
echo "🔗 You can now start the server with: python3 danbooru_prompt_server_v2_minimal.py"

# 生成训练报告
echo "📊 Generating training report..."
cat > "artifacts/training_report_$(date +%Y%m%d_%H%M%S).md" << EOF
# BGE-M3 Danbooru Training Report

## Training Configuration
- **Model**: $MODEL_NAME
- **Batch Size**: $BATCH_SIZE
- **Database**: $DB_DIR
- **Data File**: $OUTPUT_FILE
- **Training Date**: $(date)

## System Information
- **Python Version**: $(python3 --version)
- **PyTorch Version**: $(python3 -c "import torch; print(torch.__version__)")
- **CUDA Available**: $(python3 -c "import torch; print(torch.cuda.is_available())")
- **GPU Name**: $(python3 -c "import torch; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')")

## Training Results
- **Status**: ✅ Completed Successfully
- **Vector Database**: Created at $DB_DIR
- **Collections**: $(python3 -c "import chromadb; client = chromadb.PersistentClient(path='$DB_DIR'); print(len(client.list_collections()))" 2>/dev/null || echo "Unknown")

## Next Steps
1. Start the server: \`python3 danbooru_prompt_server_v2_minimal.py\`
2. Test the API endpoints
3. Integrate with your AI art generation workflow

---
Generated by run_training.sh
EOF

echo "📋 Training report saved to: artifacts/training_report_$(date +%Y%m%d_%H%M%S).md"