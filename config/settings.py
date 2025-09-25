import os
from dotenv import load_dotenv

load_dotenv()

# LangSmith配置
LANGCHAIN_API_KEY = "lse657f704d0"
LANGCHAIN_PROJECT = "MMRAG"
LANGCHAIN_TRACING_V2 = "true"
LANGCHAIN_ENDPOINT = "https://api.smith.langchain.com"

# 基础路径配置
PDF_FOLDER = "/root/MMRAG/test"  # PDF文件夹路径
PROCESS_IMAGE_OUTPUT_FOLDER = "/root/MMRAG/images"  # 图片输出文件夹路径

DB_PATH = "/root/autodl-tmp/qdrant_test5"  # 数据库路径
DB_PATH1 = "/root/autodl-tmp/qdrant_test6"  # 数据库路径
DB_NAME = "test4"  # 数据库集合名称

DB_NAME1 = "test5"  # 数据库集合名称

MODEL_CACHE_DIR = "/root/autodl-tmp/model_cache"  # 模型缓存目录

# 模型配置
MODEL_NAME = "vidore/colqwen2-v0.1"
VECTOR_SIZE = 128

# 搜索配置
DEFAULT_LIMIT = 5
DEFAULT_SCORE_THRESHOLD = 0.7
DEFAULT_BATCH_SIZE = 1
