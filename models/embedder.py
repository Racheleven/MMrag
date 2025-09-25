import torch
from colpali_engine.models import ColQwen2, ColQwen2Processor
from config.settings import MODEL_NAME, MODEL_CACHE_DIR
from fastembed import LateInteractionTextEmbedding
import numpy as np

class ColQwen2Embedder:
    """ColQwen2模型封装类（单例模式）"""
    _instance = None
    _initialized = False

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(ColQwen2Embedder, cls).__new__(cls)
        return cls._instance

    def __init__(self, model_cache_dir=MODEL_CACHE_DIR):
        if not self._initialized:
            self.device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
            self.model_name = MODEL_NAME
            self.model_cache_dir = model_cache_dir
            self.model = None
            self.processor = None
            self._init_model()
            self._initialized = True

    def _init_model(self):
        """初始化ColQwen2模型和处理器"""
        print("正在加载ColQwen2模型...")
        self.model = ColQwen2.from_pretrained(
            pretrained_model_name_or_path=self.model_name,
            torch_dtype=torch.bfloat16,
            device_map=self.device, 
            cache_dir=self.model_cache_dir
        )
        self.processor = ColQwen2Processor.from_pretrained(
            pretrained_model_name_or_path=self.model_name,
            cache_dir=self.model_cache_dir
        )
        print("✅ ColQwen2模型加载完成")

    def get_text_embedding(self, query):
        """获取文本的嵌入向量"""
        with torch.no_grad():
            text_embedding = self.processor.process_queries([query]).to(self.device)
            text_embedding = self.model(**text_embedding)
        return text_embedding[0].cpu().float().numpy().tolist()

    def get_image_embeddings(self, images):
        """获取图片的嵌入向量"""
        with torch.no_grad():
            batch_images = self.processor.process_images(images).to(self.device)
            image_embeddings = self.model(**batch_images)
        return image_embeddings 


class ColBertEmbedder:
    """ColBERT模型封装类（单例模式）
    
    使用fastembed库的LateInteractionTextEmbedding实现ColBERT嵌入。
    ColBERT使用延迟交互方式对文本进行编码，为每个token生成一个嵌入向量。
    
    特点:
    - 延迟交互: 保留token-level嵌入，而不是聚合为单一向量
    - 精确匹配: 能够捕获文本间更精细的语义匹配
    - 多向量表示: 每个文本被表示为多个向量的矩阵
    """
    _instance = None
    _initialized = False
    
    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(ColBertEmbedder, cls).__new__(cls)
        return cls._instance
    
    def __init__(self, model_name="colbert-ir/colbertv2.0", model_cache_dir=MODEL_CACHE_DIR):
        """初始化ColBERT嵌入器
        
        Args:
            model_name: ColBERT模型名称，默认为"colbert-ir/colbertv2.0"
            model_cache_dir: 模型缓存目录
        """
        if not self._initialized:
            print("正在加载ColBERT模型...")
            self.model_name = model_name
            self.model_cache_dir = model_cache_dir
            self.embedding_model = LateInteractionTextEmbedding(
                model_name, 
                cache_dir=model_cache_dir
            )
            self._initialized = True
            print(f"✅ ColBERT模型 {model_name} 加载完成")
    
    def get_text_embeddings(self, texts):
        """获取多个文本的嵌入向量矩阵
        
        Args:
            texts: 文本列表
            
        Returns:
            numpy.ndarray形式的嵌入向量矩阵列表
        """
        if isinstance(texts, str):
            texts = [texts]
            
        # 返回一个列表，每个元素是形状为(num_tokens, embedding_dim)的numpy数组
        embeddings = list(self.embedding_model.embed(texts))
        return embeddings
    
    def get_text_embedding(self, text):
        """获取单个文本的嵌入向量矩阵
        
        Args:
            text: 单个文本字符串
            
        Returns:
            numpy.ndarray形式的嵌入向量矩阵
        """
        return next(self.embedding_model.embed([text]))
    