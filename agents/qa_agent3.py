from typing_extensions import TypedDict
from agents.search_agent import SearchAgent
from agents.document import Document
from agents.prompt import TEXT_EVIDENCE_EXTRACT_PROMPT, IMAGE_EVIDENCE_EXTRACT_PROMPT
from agents.llm import myVLM
from langchain_openai import ChatOpenAI
import json
import concurrent.futures

class RetrievalResults(TypedDict):
    """检索结果的类型定义"""
    text_results: list  # 文本检索结果
    image_results: list  # 图片检索结果
    evidence: list  # 提取的证据列表

class QAAgent3:
    """简化版问答代理类，只返回检索结果"""
    
    def __init__(self, search_agent=None):
        self.search_agent = search_agent or SearchAgent()
        
        # 初始化模型
        self.model_text = ChatOpenAI(
            model="deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
            temperature=0,
            max_tokens=None,
            timeout=None,
            max_retries=2,
            api_key="sk-vzueoszaobqmmtpcbxtycvombxzwbxrbmtzllrdnahoebsui",
            base_url="https://api.siliconflow.cn/v1/",
        )
        
        self.model_image = myVLM()

    def _process_task(self, task):
        """处理单个检索任务"""
        task_type = task[0]

        if task_type == 'image':
            image_path, prompt = task[1], task[2]
            response = self.model_image.invoke(image_path, prompt)
            content = response.choices[0].message.content
            print(content)

            json_content = content.strip("```json\n").strip("```")
            parsed_data = json.loads(json_content)
            print("Parsed JSON data:", parsed_data)
            
            # 为每个证据添加图片路径前缀
            evidence_list = []
            for evidence in parsed_data.get('evidence_list', []):
                evidence_list.append(f"[图片来源: {image_path}] {evidence}")
            return {'evidence_list': evidence_list}

        elif task_type == 'text':
            prompt = task[1]
            text_content = task[3]  # 文本内容
            response = self.model_text.invoke(prompt)
            try:
                json_content = response.content.strip("```json\n").strip("```")
                parsed_data = json.loads(json_content)
                print("Parsed JSON data:", parsed_data)
                
                # 为每个证据添加文本来源前缀
                evidence_list = []
                for evidence in parsed_data.get('evidence_list', []):
                    evidence_list.append(f"[文本来源] {evidence}")
                return {'evidence_list': evidence_list}
            except json.JSONDecodeError as e:
                print("Error parsing JSON:", e)
                return {"evidence_list": []}

    def retrieve(self, query: str, document: Document, mode: str = "image") -> RetrievalResults:
        """
        从文档中检索相关内容
        
        Args:
            query (str): 查询文本
            document (Document): 文档对象
            mode (str): 检索模式，可选值："all"、"text"、"image"
            
        Returns:
            RetrievalResults: 检索结果，包含文本结果、图片结果和提取的证据
        """
        print(f"开始检索，查询：{query}，模式：{mode}")
        
        # 执行检索
        results = self.search_agent.search(
            document_obj=document,
            query=query,
            search_type=mode  # 这里固定使用image类型，因为它会同时返回文本和图片结果
        )
        
        # 准备并行任务
        tasks = []
        evidence_list = []
        
        # 根据模式添加任务
        if mode in ("image", "all"):
            for each in results["image_results"].points:
                sp = IMAGE_EVIDENCE_EXTRACT_PROMPT.format(query=query)
                image_path = each.payload['image_path']
                print(f"添加图片任务：{image_path}")
                tasks.append(('image', image_path, sp))

        if mode in ("text", "all"):
            for each in results["text_results"].points:
                text_content = document.text_blocks[each.id]
                sp = TEXT_EVIDENCE_EXTRACT_PROMPT.format(
                    query=query,
                    context=text_content
                )
                print(f"添加文本任务：text_{each.id}")
                tasks.append(('text', sp, None, text_content))

        # 使用线程池并行处理任务
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = [executor.submit(self._process_task, task) for task in tasks]
            
            for future in concurrent.futures.as_completed(futures):
                try:
                    parsed_data = future.result()
                    evidence_list.extend(parsed_data.get('evidence_list', []))
                except Exception as e:
                    print(f"处理任务时出错: {e}")

        return {
            "text_results": results["text_results"],
            "image_results": results["image_results"],
            "evidence": evidence_list
        }

    def run(self, query: str, document: Document) -> RetrievalResults:
        """
        运行检索流程
        
        Args:
            query (str): 用户查询
            document (Document): 文档对象
            
        Returns:
            RetrievalResults: 检索结果
        """
        return self.retrieve(query, document) 