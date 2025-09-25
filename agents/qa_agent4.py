from langgraph.graph import END, StateGraph,START
from typing_extensions import TypedDict
from agents.search_agent import SearchAgent
from langgraph.checkpoint.memory import MemorySaver
import os
import json
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages
from langchain_openai import ChatOpenAI
from agents.document import Document
from agents.prompt1 import INTENT_RECOGNITION_PROMPT,TEXT_EVIDENCE_EXTRACT_PROMPT,IMAGE_EVIDENCE_EXTRACT_PROMPT,CRITIC_EVIDENCE_PROMPT,ANSWER_PROMPT
from agents.llm import myVLM
import json
import concurrent.futures
import re
from agents.llm import myVLM
from langchain_openai import ChatOpenAI
import json
import concurrent.futures
import re



def ans_parser(response):
    pattern = r'<ANS>(.*?)</ANS>'
    match = re.search(pattern, response.content, re.DOTALL)
    ans_content=""

    if match:
        ans_content = match.group(1)
        print(ans_content)
    else:
        print("未找到 ANS 内容")
    return ans_content

    
def json_parser(response):
    print("==============json_parser===============")
    # 提取 JSON 内容
    json_content = response
    # 解析 JSON 数据
    try:
        parsed_data = json.loads(json_content)
        print("Parsed JSON data:", parsed_data)
    except json.JSONDecodeError as e:
        print("Error parsing JSON:", e)
    return parsed_data


class RetrievalResults(TypedDict):
    """检索结果的类型定义"""
    text_results: list  # 文本检索结果
    image_results: list  # 图片检索结果
    evidence: list  # 提取的证据列表

class QAAgent:
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
        """根据任务类型分别处理"""
        task_type = task[0]

        if task_type == 'image':
            image_path, prompt = task[1], task[2]
            response = self.model_image.invoke(image_path, prompt)
            content = response.choices[0].message.content
            print(content)
            pattern = r'<ANS>(.*?)</ANS>'
            match = re.search(pattern, content, re.DOTALL)
            parsed_data=""

            if match:
                parsed_data = match.group(1)
                print(parsed_data)
                parsed_data=f"[图片来源: {image_path}]"+parsed_data
            else:
                print("未找到 ANS 内容")

            return parsed_data

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
        
        results = self.search_agent.search(document_obj=document, query=query, search_type=mode)

        if mode in ("image", "all"):
            for each in results["image_results"].points:
                sp = IMAGE_EVIDENCE_EXTRACT_PROMPT.format(query=query)
                image_path = each.payload['image_path']
                print(image_path)
                
                tasks.append(('image', image_path, sp))

        if mode in ("text", "all"):
            for each in results["text_results"].points:
                sp = TEXT_EVIDENCE_EXTRACT_PROMPT.format(
                    query=query, context=document.text_blocks[each.id])
                tasks.append(('text', sp))

        # 用线程池并行处理
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = [executor.submit(self._process_task, task) for task in tasks]

            for future in concurrent.futures.as_completed(futures):
                try:
                    parsed_data = future.result()
                    evidence_list.append(parsed_data)
                except Exception as e:
                    print("Error during parsing:", e)

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