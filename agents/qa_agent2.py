from langgraph.graph import END, StateGraph,START
from typing_extensions import TypedDict
from agents.search_agent import SearchAgent
import os
import json
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages
from langchain_openai import ChatOpenAI
from agents.document import Document
from agents.llm import myVLM
import concurrent.futures

class GraphState(TypedDict):
    """图状态类型定义"""
    document_obj: Document
    origin_query: str  # 用户原始的查询
    messages: list  # 对话历史
    query: str  # 检索的查询
    answer_to_question: str  # 回答的问题
    mode: str  # 处理模式：text, image, all
    text_answers: list  # 文本检索的答案列表
    image_answers: list  # 图片检索的答案列表

TEXT_ANSWER_PROMPT = """你是一个专业的助手。请根据给定的文本内容，回答用户的问题。
注意：
1. 只使用给定的文本内容来回答
2. 如果文本内容不足以回答问题，请说明无法回答
3. 答案必须是一句完整的话或完整的描述，是对用户问题的完整回答，不允许是单个词或片段。

问题：{query}
文本内容：{context}

请直接给出答案："""

IMAGE_ANSWER_PROMPT = """你是一个专业的视觉问答助手。请仔细观察图片，回答用户的问题。
注意：
1. 只根据图片内容回答
2. 如果图片内容不足以回答问题，请说明无法回答
3. 答案必须是一句完整的话或完整的描述，是对用户问题的完整回答，不允许是单个词或片段。


问题：{query}

答案必须是一句完整的话或完整的描述，是对用户问题的完整回答，不允许是单个词或片段。
答案：
"""


FINAL_MERGE_PROMPT = """你是一个专业的助手。我现在给你多个来源的答案，请你进行总结和合并，给出最终的答案。

文本来源的答案：
{text_answers}

图片来源的答案：
{image_answers}

请综合以上答案，给出最终的完整回答："""

class QAAgent2:
    """直接回答的问答代理类（单例模式）"""
    _instance = None
    _initialized = False

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(QAAgent2, cls).__new__(cls)
        return cls._instance

    def __init__(self, search_agent=None):
        if not self._initialized:
            self.search_agent = search_agent or SearchAgent()
            self.workflow = self.create_workflow()

            self.model_text = ChatOpenAI(
                model="deepseek-chat",
                temperature=0,
                max_tokens=None,
                timeout=None,
                max_retries=3,
                api_key="sk-991c57bdef8a4e8aa84302be95811230",
                base_url="https://api.deepseek.com",
            )

            self.model_image = myVLM()

            print(self.workflow.get_graph().draw_mermaid())
            self._initialized = True

    def intent_recognition(self, state: GraphState):
        """简单的意图识别，直接返回查询"""
        question = state['messages'][-1]['content']
        return {
            "query": question,
            "text_answers": [],
            "image_answers": []
        }

    def retrieve_and_answer(self, state: GraphState):
        """检索并直接生成答案"""
        print("========retrieve_and_answer================")
        question = state['query']
        results = self.search_agent.search(
            document_obj=state['document_obj'], 
            query=question,
            search_type="all"
        )

        text_answers = []
        image_answers = []
        tasks = []

        # 处理文本检索结果
        if state['mode'] in ("text", "all"):
            for point in results["text_results"].points:
                context = state['document_obj'].text_blocks[point.id]
                prompt = TEXT_ANSWER_PROMPT.format(
                    query=question,
                    context=context
                )
                tasks.append(('text', prompt))

        # 处理图片检索结果
        if state['mode'] in ("image", "all"):
            for point in results["image_results"].points:
                image_path = point.payload['image_path']
                print(image_path)
                prompt = IMAGE_ANSWER_PROMPT.format(query=question)
                tasks.append(('image', image_path, prompt))

        # 并行处理所有任务
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = [executor.submit(self._process_task, task) for task in tasks]

            for future in concurrent.futures.as_completed(futures):
                try:
                    task_type, answer = future.result()
                    if task_type == 'text':
                        text_answers.append(answer)
                    else:
                        image_answers.append(answer)
                except Exception as e:
                    print(f"Error processing task: {e}")

        return {
            "text_answers": text_answers,
            "image_answers": image_answers
        }

    def _process_task(self, task):
        """处理单个任务"""
        task_type = task[0]

        if task_type == 'text':
            prompt = task[1]
            response = self.model_text.invoke(prompt)
            return 'text', response.content

        elif task_type == 'image':
            image_path, prompt = task[1], task[2]
            response = self.model_image.invoke(image_path, prompt)
            return 'image', response.choices[0].message.content

    def merge_answers(self, state: GraphState):
        """合并文本和图片的答案"""
        print("========merge_answers================")
        
        text_answers_str = "\n".join([
            f"答案 {i+1}: {answer}" 
            for i, answer in enumerate(state['text_answers'])
        ])
        
        image_answers_str = "\n".join([
            f"答案 {i+1}: {answer}" 
            for i, answer in enumerate(state['image_answers'])
        ])

        print(image_answers_str)

        prompt = FINAL_MERGE_PROMPT.format(
            text_answers=text_answers_str,
            image_answers=image_answers_str
        )

        response = self.model_text.invoke(prompt)
        return {"answer_to_question": response.content}

    def create_workflow(self):
        """创建简单的工作流"""
        workflow = StateGraph(GraphState)
        
        workflow.add_node("intent_recognition", self.intent_recognition)
        workflow.set_entry_point("intent_recognition")
        
        workflow.add_node("retrieve_and_answer", self.retrieve_and_answer)
        workflow.add_node("merge_answers", self.merge_answers)

        workflow.add_edge("intent_recognition", "retrieve_and_answer")
        workflow.add_edge("retrieve_and_answer", "merge_answers")
        workflow.add_edge("merge_answers", END)

        return workflow.compile()

    def run(self, messages: str, document: Document, mode='image'):
        """运行工作流处理问题"""
        result = self.workflow.invoke({
            "messages": messages,
            "document_obj": document,
            "mode": mode
        })
        return result["answer_to_question"] 