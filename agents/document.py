import os
import shutil
from pdf2image import convert_from_path
from tqdm import tqdm
from PIL import Image
from qdrant_client import models as qdrant_models

from config.settings import PROCESS_IMAGE_OUTPUT_FOLDER, DB_PATH, DB_PATH1, DEFAULT_BATCH_SIZE
from unstructured.partition.pdf import partition_pdf
import uuid
from datetime import datetime

class Document:
    """表示单个PDF文档的类，包含文档处理的所有相关功能和属性"""
    
    def __init__(self, pdf_path, text_db,image_db,text_embedder,image_embedder,output_folder=PROCESS_IMAGE_OUTPUT_FOLDER, process_mode="all", image_process_mode="merge"):
        """
        初始化Document对象
        
        Args:
            pdf_path (str): PDF文件的路径
            output_folder (str): 输出文件夹路径，用于存储转换后的图片
        """
        self.pdf_path = pdf_path
        self.doc_id = str(uuid.uuid4())[:8]  # 生成唯一文档ID
        self.filename = os.path.basename(pdf_path)
        
        # 设置图片输出路径
        self.output_folder = os.path.join(output_folder, self.doc_id)
        if not os.path.exists(self.output_folder):
            os.makedirs(self.output_folder)
            
        # 初始化数据
        self.text_blocks = []  # 存储提取的文本块
        self.images = []  # 存储转换的图片信息
        self.created_at = datetime.now().isoformat()
        
        # # 初始化数据库连接
        # self.text_db = QdrantManager(DB_PATH1, "text_collection")
        # self.image_db = QdrantManager(DB_PATH, "image_collection")
        
        # # 初始化embedder
        # self.text_embedder = ColBertEmbedder()
        # self.image_embedder = ColQwen2Embedder()

        self.text_db=text_db
        self.image_db=image_db

        self.text_embedder=text_embedder
        self.image_embedder=image_embedder

        self.process_mode = process_mode
        self.image_process_mode = image_process_mode

    
    def document_process(self, mode='all'):
        """处理文档的主函数，包括文本提取和图片转换"""
        try:
            mode = mode or self.process_mode
            if mode in ('all', 'text'):
                self._extract_text()
                self._process_text_blocks()
            if mode in ('all', 'image'):
                self._convert_to_images()
                if self.image_process_mode == "single":
                    self._process_images()
                elif self.image_process_mode == "merge":
                    self._process_images_merged()
            return True
        except Exception as e:
            print(f"处理文档时出错: {str(e)}")
            return False
    
    # def document_process(self, mode='all'):
    #     """处理文档的主函数，包括文本提取和图片转换"""
    #     try:
    #         if mode in ('all', 'text'):
    #             self._extract_text()
    #             self._process_text_blocks()
    #         if mode in ('all', 'image'):
    #             self._convert_to_images()
    #             self._process_images()
    #         return True
    #     except Exception as e:
    #         print(f"处理文档时出错: {str(e)}")
    #         return False

    def _extract_text(self):
        """从PDF中提取文本内容"""
        try:
            print(f"正在从 {self.filename} 提取文本...")
            raw_pdf_elements = partition_pdf(
                filename=self.pdf_path,
                extract_images_in_pdf=True,
                infer_table_structure=False,
                chunking_strategy="by_title",
                max_characters=600,
                new_after_n_chars=600,
                overlap=100,
            )
            
            for element in raw_pdf_elements:
                if "unstructured.documents.elements.CompositeElement" in str(type(element)):
                    self.text_blocks.append(element.text)
            
            print(f"成功提取 {len(self.text_blocks)} 个文本块")
            
        except Exception as e:
            print(f"提取文本时出错: {str(e)}")

    def _convert_to_images(self):
        """将PDF转换为图片"""
        try:
            images = convert_from_path(self.pdf_path)
            print(f"正在处理: {self.filename}, 共 {len(images)} 页")
            
            for page_num, image in enumerate(images):
                image_path = os.path.join(
                    self.output_folder,
                    f'page_{page_num + 1}.png'
                )
                image.save(image_path, 'PNG')
                
                self.images.append({
                    'pdf_filename': self.filename,
                    'page_num': page_num + 1,
                    'image_path': image_path
                })
            
            print(f"完成PDF转换，共处理 {len(self.images)} 页")
            
        except Exception as e:
            print(f"转换图片时出错: {str(e)}")

    def _process_text_blocks(self):
        """处理文本块并保存到文本数据库"""
        if not self.text_blocks:
            print("没有文本需要处理")
            return
            
        print("\n开始生成文本向量并保存到文本数据库...")
        
        with tqdm(total=len(self.text_blocks), desc="处理进度") as pbar:
            for i in range(0, len(self.text_blocks), DEFAULT_BATCH_SIZE):
                batch = self.text_blocks[i : i + DEFAULT_BATCH_SIZE]

                try:
                    text_embeddings = self.text_embedder.get_text_embeddings(batch)

                    points = []
                    for j, embedding in enumerate(text_embeddings):
                        points.append(
                            qdrant_models.PointStruct(
                                id=i+j,
                                vector=embedding.tolist(),
                                payload={
                                    "doc_id": self.doc_id,
                                    "created_at": self.created_at
                                },
                            )
                        )
                    
                    if self.text_db.save_points(points):
                        pbar.update(len(batch))

                except Exception as e:
                    print(f"\n处理批次 {i//DEFAULT_BATCH_SIZE + 1} 时出错: {str(e)}")
                    continue
        
        print("\n✅ 所有文字部分已保存到数据库")

    def _process_images(self):
        """处理图片并保存到图片数据库"""
        if not self.images:
            print("没有图片需要处理")
            return

        print("\n开始生成图片向量并保存到图片数据库...")
        
        with tqdm(total=len(self.images), desc="处理进度") as pbar:
            for i in range(0, len(self.images), DEFAULT_BATCH_SIZE):
                batch = self.images[i : i + DEFAULT_BATCH_SIZE]
                
                try:
                    # 读取图片
                    images = [Image.open(item['image_path']) for item in batch]
                    
                    # 生成向量
                    image_embeddings = self.image_embedder.get_image_embeddings(images)
                    
                    points = []
                    for j, embedding in enumerate(image_embeddings):
                        points.append(
                            qdrant_models.PointStruct(
                                id=i+j,
                                vector=embedding.tolist(),
                                payload={
                                    "doc_id": self.doc_id,
                                    "pdf_filename": batch[j]["pdf_filename"],
                                    "page_num": batch[j]["page_num"],
                                    "image_path": batch[j]["image_path"],
                                    "created_at": self.created_at
                                },
                            )
                        )
                    
                    if self.image_db.save_points(points):
                        pbar.update(len(batch))
                    
                except Exception as e:
                    print(f"\n处理批次 {i//DEFAULT_BATCH_SIZE + 1} 时出错: {str(e)}")
                    continue

        print("\n✅ 所有图片已保存到数据库")

    def _process_images_merged(self):
        """将相邻两张图片拼接后生成embedding并保存到图片数据库"""
        if not self.images or len(self.images) < 2:
            print("没有足够的图片进行拼接处理")
            return

        print("\n开始拼接图片并生成图片向量保存到图片数据库...")

        merged_image_infos = []
        for i in range(len(self.images) - 1):
            img1_path = self.images[i]['image_path']
            img2_path = self.images[i+1]['image_path']
            im1 = Image.open(img1_path)
            im2 = Image.open(img2_path)
            # 竖直拼接
            dst = Image.new('RGB', (im1.width, im1.height + im2.height))
            dst.paste(im1, (0, 0))
            dst.paste(im2, (0, im1.height))
            merged_path = os.path.join(self.output_folder, f'merged_{i+1}_{i+2}.png')
            dst.save(merged_path)
            merged_image_infos.append({
                'pdf_filename': self.filename,
                'page_num': f"{self.images[i]['page_num']}_{self.images[i+1]['page_num']}",
                'image_path': merged_path
            })

        # 批量处理拼接图片
        with tqdm(total=len(merged_image_infos), desc="拼接图片处理进度") as pbar:
            for i in range(0, len(merged_image_infos), DEFAULT_BATCH_SIZE):
                batch = merged_image_infos[i : i + DEFAULT_BATCH_SIZE]
                try:
                    images = [Image.open(item['image_path']) for item in batch]
                    image_embeddings = self.image_embedder.get_image_embeddings(images)
                    points = []
                    for j, embedding in enumerate(image_embeddings):
                        points.append(
                            qdrant_models.PointStruct(
                                id=i+j,
                                vector=embedding.tolist(),
                                payload={
                                    "doc_id": self.doc_id,
                                    "pdf_filename": batch[j]["pdf_filename"],
                                    "page_num": batch[j]["page_num"],
                                    "image_path": batch[j]["image_path"],
                                    "created_at": self.created_at
                                },
                            )
                        )
                    if self.image_db.save_points(points):
                        pbar.update(len(batch))
                except Exception as e:
                    print(f"\n处理拼接图片批次 {i//DEFAULT_BATCH_SIZE + 1} 时出错: {str(e)}")
                    continue

        print("\n✅ 所有拼接图片已保存到数据库")

    def delete(self):
        """删除文档相关的所有资源"""
        try:
            # 1. 删除图片文件夹
            if os.path.exists(self.output_folder):
                shutil.rmtree(self.output_folder)
                print(f"已删除图片文件夹: {self.output_folder}")
            
            # 2. 从文本数据库中删除记录
            self.text_db.delete_by_filter(self.doc_id,"text_collection")
            print("已从文本数据库删除记录")
            
            # 3. 从图片数据库中删除记录
            self.image_db.delete_by_filter(self.doc_id,"image_collection")
            print("已从图片数据库删除记录")
            
            return True
        except Exception as e:
            print(f"删除文档资源时出错: {str(e)}")
            return False

    # def __del__(self):
    #     """析构函数，确保关闭数据库连接"""
    #     if hasattr(self, 'text_db'):
    #         self.text_db.close()
    #     if hasattr(self, 'image_db'):
    #         self.image_db.close() 