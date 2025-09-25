import os
from pdf2image import convert_from_path
from tqdm import tqdm
from PIL import Image
from qdrant_client import models as qdrant_models
from models.embedder import ColQwen2Embedder, ColBertEmbedder
from models.database import QdrantManager
from config.settings import PDF_FOLDER, OUTPUT_FOLDER, DB_PATH, DB_PATH1,DB_NAME, DEFAULT_BATCH_SIZE
from unstructured.partition.pdf import partition_pdf
import uuid
from datetime import datetime

class DocumentProcessor:
    """文档处理类，支持PDF转图片和OCR文本提取"""
    def __init__(self, pdf_folder=PDF_FOLDER, output_folder=OUTPUT_FOLDER):
        self.pdf_folder = pdf_folder
        self.output_folder = output_folder
        # 初始化两个embedder
        self.image_embedder = ColQwen2Embedder()
        self.text_embedder = ColBertEmbedder()
        # 初始化两个db_manager
        self.image_db = QdrantManager(DB_PATH, "image_collection")  # 图片向量集合
        self.text_db = QdrantManager(DB_PATH1, "text_collection")    # 文本向量集合

    def close(self):
        """关闭所有数据库连接"""
        if self.image_db:
            self.image_db.close()
        if self.text_db:
            self.text_db.close()

    def convert_pdf2image(self, pdf_files):
        """将PDF转换为图片"""
        # 创建输出文件夹
        if not os.path.exists(self.output_folder):
            os.makedirs(self.output_folder)
            print(f"创建输出文件夹: {self.output_folder}")

        all_images = []
        pdf_files = [f for f in os.listdir(self.pdf_folder) if f.endswith('.pdf')]
        
        if not pdf_files:
            print(f"警告: 在 {self.pdf_folder} 中没有找到PDF文件")
            return []

        print(f"找到 {len(pdf_files)} 个PDF文件")
        
        for filename in tqdm(pdf_files, desc="处理PDF文件"):
            pdf_path = os.path.join(self.pdf_folder, filename)
            try:
                images = convert_from_path(pdf_path)
                print(f"正在处理: {filename}, 共 {len(images)} 页")
                
                for page_num, image in enumerate(images):
                    image_path = os.path.join(self.output_folder, f'{filename}_page_{page_num + 1}.png')
                    image.save(image_path, 'PNG')
                    
                    all_images.append({
                        'pdf_filename': filename,
                        'page_num': page_num + 1,
                        'image_path': image_path
                    })
                    
            except Exception as e:
                print(f"处理文件 {filename} 时出错: {str(e)}")
                continue

        print(f"完成PDF转换，共处理 {len(all_images)} 页")
        return all_images

    def extract_text_from_pdf(self, pdf_path):
        """从PDF中提取文本内容
        
        Args:
            pdf_path: PDF文件路径
            
        Returns:
            list: 包含文本块信息的列表，每个元素是一个字典
                {
                    'text': 提取的文本内容,
                    'page_num': 页码,
                    'pdf_filename': PDF文件名
                }
        """
        try:
            print(f"正在从 {os.path.basename(pdf_path)} 提取文本...")
            # Get elements
            raw_pdf_elements = partition_pdf(
                filename=pdf_path,
                extract_images_in_pdf=True,
                infer_table_structure=False,
                chunking_strategy="by_title",
                max_characters=600,
                new_after_n_chars=600,
                overlap=100,
            )
            
            # 处理提取的文本元素
            chunk_lists = []
            
            for element in raw_pdf_elements:
                if "unstructured.documents.elements.CompositeElement" in str(type(element)):
                    chunk_lists.append(element.text)
                

            return chunk_lists
            
        except Exception as e:
            print(f"提取文本时出错: {str(e)}")
            return []

    def generate_document_id(self,):
        """为文档生成唯一ID
        
        Args:

        Returns:
            str: 唯一文档ID
        """
        # 使用UUID和时间戳生成唯一ID
        # timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        # unique_id = str(uuid.uuid4())[:8]
        # return f"doc_{timestamp}_{unique_id}"

        return str(uuid.uuid4())[:8]
    
    def process_text_blocks(self, text_blocks, doc_id):
        """处理文本块并保存到文本数据库"""
        if not text_blocks:
            print("没有文本需要处理")
            return
            
        print("\n开始生成文本向量并保存到文本数据库...")
        
        #生成文本向量
        with tqdm(total=len(text_blocks), desc="处理进度") as pbar:
            for i in range(0, len(text_blocks), DEFAULT_BATCH_SIZE):
                batch = text_blocks[i : i + DEFAULT_BATCH_SIZE]

                try:
                    text_embeddings = self.text_embedder.get_text_embeddings(batch)

                    # 保存到文本数据库
                    points = []
                    for j, embedding in enumerate(text_embeddings):
                        points.append(
                            qdrant_models.PointStruct(
                                id=i+j,
                                vector=embedding.tolist(),
                                payload={
                                    "doc_id": doc_id,
                                    # "pdf_filename": batch[j]["pdf_filename"],
                                    "created_at": datetime.now().isoformat()
                                },
                            )
                        )
                    # 保存到文本数据库
                    if self.text_db.save_points(points):
                        pbar.update(DEFAULT_BATCH_SIZE)

                except Exception as e:
                    print(f"\n处理批次 {i//DEFAULT_BATCH_SIZE + 1} 时出错: {str(e)}")
                    continue
        print("\n✅ 所有文字部分已保存到数据库")
            
                    

    def process_images(self, all_images, doc_id):
        """处理图片并保存到图片数据库"""
        if not all_images:
            print("没有图片需要处理")
            return

        print("\n开始生成图片向量并保存到图片数据库...")
        
        with tqdm(total=len(all_images), desc="处理进度") as pbar:
            for i in range(0, len(all_images), DEFAULT_BATCH_SIZE):
                batch = all_images[i : i + DEFAULT_BATCH_SIZE]
                
                try:
                    # 读取图片
                    images = [Image.open(item['image_path']) for item in batch]
                    
                    # 生成向量
                    image_embeddings = self.image_embedder.get_image_embeddings(images)
                    
                    # 准备数据点
                    points = []
                    for j, embedding in enumerate(image_embeddings):
                        points.append(
                            qdrant_models.PointStruct(
                                id=i+j,
                                vector=embedding.tolist(),
                                payload={
                                    "doc_id": doc_id,
                                    "pdf_filename": batch[j]["pdf_filename"],
                                    "page_num": batch[j]["page_num"],
                                    "image_path": batch[j]["image_path"],
                                    "created_at": datetime.now().isoformat()
                                },
                            )
                        )
                    
                    # 保存到图片数据库
                    if self.image_db.save_points(points):
                        pbar.update(DEFAULT_BATCH_SIZE)
                    
                except Exception as e:
                    print(f"\n处理批次 {i//DEFAULT_BATCH_SIZE + 1} 时出错: {str(e)}")
                    continue

        print("\n✅ 所有图片已保存到数据库")

    def process_documents(self):
        """处理文档的主函数"""
        try:
            # 检查PDF文件夹
            if not os.path.exists(self.pdf_folder):
                print(f"错误: PDF文件夹 {self.pdf_folder} 不存在")
                return
            
            pdf_files = [f for f in os.listdir(self.pdf_folder) if f.endswith('.pdf')]
            if not pdf_files:
                print(f"警告: 在 {self.pdf_folder} 中没有找到PDF文件")
                return
                
            print(f"找到 {len(pdf_files)} 个PDF文件")
            
            # 处理每个PDF文件
            for filename in pdf_files:
                # 为每个文档生成唯一ID
                pdf_path="/root/MMRAG/05-03-18-political-release.pdf"
                doc_id = self.generate_document_id()
                # pdf_path = os.path.join(self.pdf_folder, filename)
                
                print(f"处理文档 {pdf_path} ")
                
                # 1. 提取文本
                text_blocks = self.extract_text_from_pdf(pdf_path)
                print("extract text done")
                self.process_text_blocks(text_blocks, doc_id)
                print("process text done")
                
                # 2. 转换为图片并处理
                images = self.convert_pdf2image(pdf_path)
                print("convert pdf2image done")
                self.process_images(images, doc_id)
                print("process images done")
        finally:
            self.close()

    def __del__(self):
        """析构函数，确保关闭所有连接"""
        self.close() 