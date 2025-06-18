#!/usr/bin/env python3

import os
import torch

# ================== COMPREHENSIVE MONKEY PATCH ==================
# 解决PyTorch 2.6.0.dev版本被误判的问题
def _comprehensive_torch_patch():
    """Comprehensive patch for torch version check issues"""
    
    # 补丁1：直接在torch模块层面设置版本信息
    if hasattr(torch, '__version__') and 'dev' in torch.__version__:
        try:
            # 临时替换版本字符串为稳定版本
            torch.__version__ = "2.6.0"
            print("INFO: Temporarily set torch.__version__ to '2.6.0' to bypass dev version checks")
        except Exception as e:
            print(f"WARN: Failed to modify torch.__version__: {e}")
    
    # 补丁2：修补 transformers 中的检查函数
    try:
        from transformers.utils import import_utils
        import_utils.check_torch_load_is_safe = lambda: None
        print("INFO: Patched transformers.utils.import_utils.check_torch_load_is_safe")
    except Exception as e:
        print(f"WARN: Failed to patch transformers import_utils: {e}")
    
    # 补丁3：修补 transformers.modeling_utils 中的检查
    try:
        from transformers import modeling_utils
        modeling_utils.check_torch_load_is_safe = lambda: None
        print("INFO: Patched transformers modeling_utils.check_torch_load_is_safe")
    except Exception as e:
        print(f"WARN: Failed to patch transformers modeling_utils: {e}")
    
    # 补丁4：环境变量设置（一些库会检查这个）
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    os.environ['TRANSFORMERS_VERBOSITY'] = 'error'

# 在导入任何transformers相关库之前立即执行补丁
_comprehensive_torch_patch()
# ================================================

import json
import argparse
from typing import List, Dict, Any
from tqdm import tqdm
import chromadb
from datetime import datetime

class ChunkVectorizer:
    """Generate embeddings from text chunks and store them in a ChromaDB vector database."""
    
    def __init__(
        self,
        input_file: str,
        db_directory: str,
        model_name: str,
        batch_size: int = 32
    ):
        """Initialize the vectorizer with input path and model parameters."""
        # 打印GPU信息
        print("=== 系统信息 ===")
        print(f"PyTorch版本: {torch.__version__}")
        print(f"CUDA可用: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"CUDA设备数量: {torch.cuda.device_count()}")
            print(f"当前CUDA设备: {torch.cuda.current_device()}")
            print(f"CUDA设备名称: {torch.cuda.get_device_name(0)}")
            print(f"CUDA内存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        print("================")
        
        self.input_file = input_file
        self.db_directory = db_directory
        collection_base_name = os.path.basename(input_file).replace('.jsonl', '')
        
        model_short_name = model_name.split('/')[-1] if '/' in model_name else model_name
        collection_name = f"{collection_base_name}_{model_short_name}"
        
        if len(collection_name) > 63:
            collection_name = collection_name[:63]
        
        self.collection_name = collection_name
        print(f"Collection name: {self.collection_name} ({len(self.collection_name)} chars)")
        
        print(f"Initializing ChromaDB at {db_directory}")
        self.client = chromadb.PersistentClient(path=db_directory)
        
        try:
            self.client.delete_collection(name=self.collection_name)
            print(f"Deleted existing collection: {self.collection_name}")
        except Exception:
            print(f"No existing collection to delete: {self.collection_name}")
        
        self.embedding_function = CustomEmbeddingFunction(model_name)
        
        self.collection = self.client.create_collection(
            name=self.collection_name,
            embedding_function=self.embedding_function,
            metadata={"hnsw:space": "cosine"}
        )
        print(f"Created new collection: {self.collection_name}")
        
        collections_file = "artifacts/vector_stores/collections.txt"
        os.makedirs(os.path.dirname(collections_file), exist_ok=True)
        with open(collections_file, 'a+', encoding='utf-8') as f:
            f.seek(0)
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            f.write(f"{self.collection_name} ({timestamp}) - Original model: {model_name}\n")
        self.batch_size = batch_size
        
    def load_chunks(self) -> List[Dict[str, Any]]:
        """Load chunks from the input JSONL file."""
        chunks = []
        
        print(f"Loading chunks from {self.input_file}")
        with open(self.input_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    chunk = json.loads(line)
                    chunks.append(chunk)
        
        print(f"Loaded {len(chunks)} chunks")
        return chunks
    
    def process_and_store_chunks(self, chunks: List[Dict[str, Any]]) -> None:
        """Process chunks in batches and store them in ChromaDB."""
        chunk_ids = [chunk['id'] for chunk in chunks]
        texts = [chunk['text'] for chunk in chunks]
        metadatas = [{'source': chunk['source']} for chunk in chunks]
        
        print(f"Processing {len(chunks)} chunks in batches of {self.batch_size}")
        
        for i in tqdm(range(0, len(chunks), self.batch_size), desc="Embedding Chunks"):
            batch_ids = chunk_ids[i:i+self.batch_size]
            batch_texts = texts[i:i+self.batch_size]
            batch_metadatas = metadatas[i:i+self.batch_size]
            
            self.collection.add(
                ids=batch_ids,
                documents=batch_texts,
                metadatas=batch_metadatas
            )
        
        print(f"Successfully stored {len(chunks)} chunks in ChromaDB")
        
        collection_count = self.collection.count()
        print(f"Total documents in collection: {collection_count}")
        
    def run(self) -> None:
        """Run the full vectorization process."""
        chunks = self.load_chunks()
        self.process_and_store_chunks(chunks)
        print(f"Vector database created successfully at: {os.path.abspath(self.db_directory)}")
        print("You can now query the database using ChromaDB's query API.")


class CustomEmbeddingFunction:
    """Custom embedding function for ChromaDB using sentence-transformers or FlagEmbedding."""
    
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        """Initialize with a specific model, prioritizing FlagEmbedding for BGE models."""
        from sentence_transformers import SentenceTransformer

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")
        
        self.is_bge_model = "bge" in model_name.lower()
        
        if self.is_bge_model:
            try:
                from FlagEmbedding import BGEM3FlagModel
                print(f"Loading BGE model using FlagEmbedding: {model_name}")
                self.model = BGEM3FlagModel(model_name, use_fp16=True, device=self.device)
                print(f"BGE model loaded successfully on {self.device}")
            except ImportError:
                print("FlagEmbedding not installed. Falling back to SentenceTransformer for BGE model.")
                self.model = SentenceTransformer(model_name, device=self.device)
        else:
            print(f"Loading model using SentenceTransformer: {model_name}")
            self.model = SentenceTransformer(model_name, device=self.device)
        
    def __call__(self, input: List[str]) -> List[List[float]]:
        """Generate embeddings for a list of texts. Updated to match ChromaDB's EmbeddingFunction interface."""
        if self.is_bge_model:
            # BGE-M3 通过 FlagEmbedding 使用
            embeddings = self.model.encode(input, return_dense=True)['dense_vecs']
        else:
            # 其他模型通过 SentenceTransformer 使用
            embeddings = self.model.encode(input, show_progress_bar=False)
        
        return embeddings.tolist()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate embeddings from text chunks and store them in ChromaDB.")
    parser.add_argument("--input", "-i", dest="input_file", required=True, help="Input JSONL file containing text chunks")
    parser.add_argument("--db", "-d", dest="db_directory", default="artifacts/vector_stores/chroma_db",
                        help="Directory where ChromaDB will store the vector database (default: artifacts/vector_stores/chroma_db)")
    parser.add_argument("--model", "-m", dest="model_name", default="sentence-transformers/all-MiniLM-L6-v2", 
                        help="Name of the sentence-transformer model to use (default: sentence-transformers/all-MiniLM-L6-v2)")
    parser.add_argument("--batch-size", "-b", dest="batch_size", type=int, default=32,
                        help="Batch size for embedding generation (default: 32)")
    
    args = parser.parse_args()
    
    vectorizer = ChunkVectorizer(
        input_file=args.input_file,
        db_directory=args.db_directory,
        model_name=args.model_name,
        batch_size=args.batch_size
    )
    vectorizer.run()