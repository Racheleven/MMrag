from models.embedder import ColQwen2Embedder, ColBertEmbedder
from models.database import QdrantManager

from config.settings import DB_NAME, DB_PATH,DB_NAME1,DB_PATH1

class SearchAgent:
    """搜索代理类（单例模式）"""
    _instance = None
    _initialized = False

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(SearchAgent, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        if not self._initialized:
            # # 初始化两个embedder
            # self.image_embedder = ColQwen2Embedder()
            # self.text_embedder = ColBertEmbedder()
            
            # # 初始化两个数据库连接
            # self.image_db = QdrantManager(DB_PATH, "image_collection")
            # self.text_db = QdrantManager(DB_PATH1, "text_collection")
            
            self._initialized = True

    def search_images(self, document_obj,query: str, limit: int = 5, score_threshold: float = 0.5):
        """搜索相关图片
        
        Args:
            query: 查询文本
            limit: 返回结果数量
            score_threshold: 相似度阈值
            
        Returns:
            list: 匹配的图片结果列表
        """
        # 初始化两个embedder
        image_embedder = document_obj.image_embedder
        # 初始化两个数据库连接
        image_db = document_obj.image_db

        try:
            # 使用ColQwen2生成查询向量
            query_vector = image_embedder.get_text_embedding(query)
            
            # 在图片集合中搜索
            search_results = image_db.search(
                document_obj.doc_id,
                query_vector, 
                limit=limit, 
                score_threshold=score_threshold
            )
            
            
            return search_results
        except Exception as e:
            print(f"图片搜索出错: {str(e)}")
            return []

    def search_texts(self,document_obj,query: str, limit: int = 5, score_threshold: float = 0.5):
        """搜索相关文本
        
        Args:
            query: 查询文本
            limit: 返回结果数量
            score_threshold: 相似度阈值
            
        Returns:
            list: 匹配的文本结果列表
        """
        # 初始化两个embedder
        text_embedder = document_obj.text_embedder
        # 初始化两个数据库连接
        text_db = document_obj.text_db

        try:
            # 使用ColBERT生成查询向量
            query_vector = text_embedder.get_text_embedding(query)
            print(query_vector)
            # query_vector = query_embedding # 取平均得到单个向量
            
            # 在文本集合中搜索
            search_results = text_db.search(
                document_obj.doc_id,
                query_vector.tolist(), 
                limit=limit, 
                score_threshold=score_threshold
            )
            
            return search_results
        except Exception as e:
            print(f"文本搜索出错: {str(e)}")
            return []

    def search(self, document_obj, query: str, limit: int = 2, score_threshold: float = 0.5, search_type="all"):
        """统一搜索接口
        
        Args:
            query: 查询文本
            limit: 每种类型返回的结果数量
            score_threshold: 相似度阈值
            search_type: 搜索类型，可选值："all"、"image"、"text"
            
        Returns:
            dict: 包含图片和文本搜索结果的字典
        """
        results = {
            "image_results": [],
            "text_results": []
        }
        
        try:
            if search_type in ["all", "image"]:
                results["image_results"] = self.search_images(
                    document_obj,
                    query, 
                    limit=3, 
                    score_threshold=score_threshold
                )
                
            if search_type in ["all", "text"]:
                results["text_results"] = self.search_texts(
                    document_obj,
                    query, 
                    limit=5, 
                    score_threshold=score_threshold
                )
                
            return results
        except Exception as e:
            print(f"搜索出错: {str(e)}")
            return results

    # def close(self):
    #     """关闭数据库连接"""
    #     if hasattr(self, 'image_db'):
    #         self.image_db.close()
    #     if hasattr(self, 'text_db'):
    #         self.text_db.close()

    # def __del__(self):
    #     """析构函数"""
    #     self.close()