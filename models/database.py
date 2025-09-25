import time
from qdrant_client import models as qdrant_models, QdrantClient
from config.settings import VECTOR_SIZE, DEFAULT_LIMIT, DEFAULT_SCORE_THRESHOLD

class QdrantManager:
    """Qdrant数据库管理类（单例模式）"""
    _instances = {}
    _initialized = {}

    def __new__(cls, db_path, collection_name):
        key = (db_path, collection_name)
        if key not in cls._instances:
            cls._instances[key] = super(QdrantManager, cls).__new__(cls)
        return cls._instances[key]

    def __init__(self, db_path, collection_name):
        key = (db_path, collection_name)
        if key not in self._initialized:
            self.db_path = db_path
            self.collection_name = collection_name
            self.client = None
            self._init_client()
            self._initialized[key] = True

    def _init_client(self):
        """初始化Qdrant客户端"""
        print("正在连接Qdrant数据库...")
        self.client = QdrantClient(path=self.db_path)
        self._ensure_collection_exists()
        print("✅ Qdrant数据库连接成功")

    def close(self):
        """关闭数据库连接"""
        if self.client:
            self.client.close()
            self.client = None
            print("✅ 数据库连接已关闭")

    def check_database(self):
        """检查数据库内容"""
        try:
            # 获取集合信息
            collection_info = self.client.get_collection(collection_name=self.collection_name)
            print("\n数据库集合信息:")
            print(f"集合名称: {self.collection_name}")
            print(f"向量维度: {collection_info.config.params.vectors.size}")
            print(f"向量数量: {collection_info.points_count}")
            
            # 获取一些示例点
            if collection_info.points_count > 0:
                print("\n示例数据点:")
                points = self.client.scroll(
                    collection_name=self.collection_name,
                    limit=min(3, collection_info.points_count)
                )[0]
                
                for point in points:
                    print(f"\nID: {point.id}")
                    print(f"来源文档: {point.payload['pdf_filename']}")
                    print(f"页码: {point.payload['page_num']}")
                    print(f"图片路径: {point.payload['image_path']}")
            else:
                print("\n数据库为空，没有存储任何图片向量")
                
        except Exception as e:
            print(f"检查数据库时出错: {str(e)}")

    def _ensure_collection_exists(self):
        """确保集合存在，如果不存在则创建"""
        try:
            collection_info = self.client.get_collection(collection_name=self.collection_name)
            print(f"使用现有集合: {self.collection_name}")
        except Exception:
            print(f"创建新集合: {self.collection_name}")
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=qdrant_models.VectorParams(
                    size=VECTOR_SIZE,
                    distance=qdrant_models.Distance.COSINE,
                    on_disk=True,
                    multivector_config=qdrant_models.MultiVectorConfig(
                    comparator=qdrant_models.MultiVectorComparator.MAX_SIM
                ),
                )
            )

    def search(self, doc_id, query_vector, limit=DEFAULT_LIMIT, score_threshold=DEFAULT_SCORE_THRESHOLD):
        """搜索相似向量"""
        try:
            start_time = time.time()
            results = self.client.query_points(
                collection_name=self.collection_name,
                query=query_vector,
                limit=limit,
                query_filter=qdrant_models.Filter(
                    must=[
                        qdrant_models.FieldCondition(
                            key="doc_id",
                            match=qdrant_models.MatchValue(
                                value=doc_id,
                            ),
                        )
                    ]
                ),
            )
            search_time = time.time() - start_time
            print(f"搜索完成，用时 {search_time:.3f} 秒")
            print(results)
            print(f"找到 {len(results.points)} 个结果")
            return results
        except Exception as e:
            print(f"❌ 搜索时出错: {str(e)}")
            return None
        
    def delete_by_filter(self,doc_id,db_name):
        # 定义过滤条件

        self.client.delete(
            collection_name=db_name,
            points_selector=qdrant_models.FilterSelector(
                filter=qdrant_models.Filter(
                    must=[
                        qdrant_models.FieldCondition(
                            key="doc_id",
                            match=qdrant_models.MatchValue(value=doc_id),
                        ),
                    ],
                )
            ),
        )

        print("删除完成！")

    def save_points(self, points):
        """保存向量点到数据库"""
        try:
            self.client.upsert(
                collection_name=self.collection_name,
                points=points
            )
            return True
        except Exception as e:
            print(f"保存到数据库时出错: {str(e)}")
            return False

    def __del__(self):
        """析构函数，确保关闭数据库连接"""
        self.close() 