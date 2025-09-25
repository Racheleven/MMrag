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

os.environ['LANGSMITH_TRACING'] = 'true'
os.environ['LANGSMITH_ENDPOINT'] = "https://api.smith.langchain.com"
os.environ['LANGSMITH_API_KEY'] = "lsv9b"
os.environ['LANGSMITH_PROJECT'] = "final_v2_0513_graduation"


class GraphState(TypedDict):
    """图状态类型定义"""
    document_obj: Document
    origin_query: str #用户原始的查询
    messages: list  # 对话历史
    query: str  # 检索的查询
    max_try: int  # 最大尝试次数
    answer_to_question: str  # 回答的问题
    missing_query: str  # 缺失的信息
    mode:str

    evidence: list

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
        parsed_data=""
    return parsed_data


class QAAgent:
    """问答代理类（单例模式）"""
    _instance = None
    _initialized = False

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(QAAgent, cls).__new__(cls)
        return cls._instance

    def __init__(self, search_agent=None):
        if not self._initialized:
            self.config = {"configurable": {"thread_id": "1"}}

            self.search_agent = search_agent or SearchAgent()
            self.workflow = self.create_workflow()

            self.model_text= ChatOpenAI(
                model="deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
                temperature=0,
                max_tokens=None,
                timeout=None,
                max_retries=2,
                api_key="sk-vzueoebsui",  # if you prefer to pass api key in directly instaed of using env vars
                base_url="https://api.siliconflow.cn/v1/",
            )

            self.model_deepseek_v3= ChatOpenAI(
                model="deepseek-chat",
                temperature=0,
                max_tokens=None,
                timeout=None,
                max_retries=3,
                api_key="sk11230",  # if you prefer to pass api key in directly instaed of using env vars
                base_url="https://api.deepseek.com",
            )

            self.model_deepseek_r1= ChatOpenAI(
                model="deepseek-reasoner",
                temperature=0,
                max_tokens=None,
                timeout=None,
                max_retries=3,
                api_key="30",  # if you prefer to pass api key in directly instaed of using env vars
                base_url="https://api.deepseek.com",
            )

            self.model_image= myVLM()


            # 绘制workflow
            print(self.workflow.get_graph().draw_mermaid())

            self._initialized = True
    
    
    def intent_recognition(self,state:GraphState):
        print("========intent_recognition================")
        #('user', '我想知道她准确的岁数是多少岁，帮我查一下')
        question=state['messages'][-1]['content']
        # _ , question=state['messages'][-1]
        msg=state['messages'][:-1]

        # 将历史记录转换为字符串，格式为对话文本
        # context = "\n".join(msg)
        context = "\n".join(
            f"{each['role']}: {each['content'] or ''}" for each in msg
        )

        sp=INTENT_RECOGNITION_PROMPT.format(query=question,context=context)
        print(sp)
        response=self.model_deepseek_v3.invoke(sp)
        print(response)
        parsed_data=json_parser(ans_parser(response))
        print(parsed_data)
        
        if parsed_data['router']=='answer':
            return {"answer_to_question":parsed_data['answer']}
        else:
            if parsed_data['query']:
                return {"origin_query":question,"query":parsed_data['query'],'answer_to_question':""}
            else:
                return {"origin_query":question,"query":question,'answer_to_question':""}
                      
    def retrieve(self, state: GraphState):
        print("========retrieve================")
        """从数据库中检索相关图片"""
        print("进入了 search 节点,查询为:", state['query'])

        max_try= state['max_try'] 
        if state['max_try'] >= 3:
            return {"answer_to_question": "没有找到足够的信息回答问题。"}

        key = "missing_query"
        question = state['query'] if key not in state or not state[key] else state[key]

        results = self.search_agent.search(document_obj=state['document_obj'], query=question, search_type=state['mode'])

        key = "evidence"
        if key not in state or not state[key]:
            evidence_lists = []
        else:
            evidence_lists = state["evidence"]

        tasks = []

        if state['mode'] in ("image", "all"):
            for each in results["image_results"].points:
                sp = IMAGE_EVIDENCE_EXTRACT_PROMPT.format(query=question)
                image_path = each.payload['image_path']
                print(image_path)
                tasks.append(('image', image_path, sp))

        if state['mode'] in ("text", "all"):
            for each in results["text_results"].points:
                sp = TEXT_EVIDENCE_EXTRACT_PROMPT.format(
                    query=question, context=state['document_obj'].text_blocks[each.id])
                tasks.append(('text', sp))

        # 用线程池并行处理
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = [executor.submit(self._process_task, task) for task in tasks]

            for future in concurrent.futures.as_completed(futures):
                try:
                    parsed_data = future.result()
                    evidence_lists.append(parsed_data)
                except Exception as e:
                    print("Error during parsing:", e)

        return {"evidence": evidence_lists,'max_try': max_try+1 }

    def _process_task(self, task):
        """根据任务类型分别处理"""
        task_type = task[0]

        if task_type == 'image':
            image_path, prompt = task[1], task[2]
            response = self.model_image.invoke(image_path, prompt)
            print("证据抽取环节哈哈哈哈哈哈哈哈哈")
            content = response.choices[0].message.content
            print(content)
            pattern = r'<ANS>(.*?)</ANS>'
            match = re.search(pattern, content, re.DOTALL)
            parsed_data=""

            if match:
                parsed_data = match.group(1)
                print(parsed_data)
            else:
                print("未找到 ANS 内容")

            # json_content = content.strip("```json\n").strip("```")
            # parsed_data = json.loads(json_content)
            # print("Parsed JSON data:", parsed_data)
            return parsed_data

        elif task_type == 'text':
            prompt = task[1]
            response = self.model_text.invoke(prompt)
            parsed_data = ans_parser(response)
            return parsed_data
    

    def critic_evidence(self,state:GraphState):
        print("========critic_evidence================")
        #分析是否evidence是否对回答问题有用。是否有遗漏信息
        missing_query=None

        if state["answer_to_question"]:
            return {"missing_query":missing_query}

        evidence = state["evidence"]
        evidence_str = '\n'.join([f"证据{idx + 1}：{item}" for idx, item in enumerate(evidence)])
        print(evidence_str)

        sp=CRITIC_EVIDENCE_PROMPT.format(evidence=evidence_str,query=state["query"])


        response=self.model_deepseek_v3.invoke(sp)
        parsed_data=json_parser(ans_parser(response))


        if parsed_data["router"]=="search":
            missing_query= parsed_data["query"]

        return {"missing_query":missing_query}

    def answer(self, state: GraphState):
        print("========answer===============")
        if state["answer_to_question"]:
            return {"answer_to_question": state["answer_to_question"]}
        else:
            evidence = state["evidence"]
            evidence_str = '\n'.join([f"证据{idx + 1}：{item}" for idx, item in enumerate(evidence)])

            sp=ANSWER_PROMPT.format(evidence=evidence_str,query=state["query"])
            response=self.model_deepseek_r1.invoke(sp)
            final_answer=ans_parser(response)

            return {"answer_to_question": final_answer}
    
    def first_router(self,state:GraphState):
        print("========first_router===============")
        key="answer_to_question"
        print(state)
        if key not in state or not state[key]:
            return "retrieve"
        else:
            return "answer"
        

    def second_router(self,state:GraphState):
        print("========second_router================")
        key="missing_query"
        if key not in state or not state[key]:
            return "answer"
        else:
            return "retrieve"
        
    def create_workflow(self):
        
        """创建检索和回答工作流"""
        workflow = StateGraph(GraphState)
        
        # 添加节点

        workflow.add_node("intent_recognition", self.intent_recognition)
        workflow.set_entry_point("intent_recognition")

        workflow.add_node("retrieve", self.retrieve)
        workflow.add_node("answer", self.answer)
        workflow.add_node("critic",self.critic_evidence)


        # 添加条件边
        workflow.add_conditional_edges("intent_recognition", self.first_router,
                                        {
                                           "retrieve": "retrieve",
                                            "answer":  "answer",
                                        })
        workflow.add_edge("retrieve","critic")
        workflow.add_conditional_edges("critic",self.second_router,
                                        {
                                           "retrieve": "retrieve",
                                            "answer":  "answer",
                                        })

        workflow.add_edge("answer", END)
        

        return workflow.compile()

    def run(self, messages: str, document: Document, mode: str = "both"):
        """运行工作流处理问题
        
        Args:
            messages (str): 对话历史
            document (Document): 文档对象
            mode (str, optional): 检索模式. 可选值: "text"(文本检索), "image"(图片检索), "both"(双检索). 默认为 "both"
        """
        result = self.workflow.invoke({
            "messages": messages,
            "document_obj": document,
            "max_try": 0,
            "mode": mode
        })
        return result["answer_to_question"]
    
# if __name__ == "__main__":
#     # 测试用的PDF文件路径
#     pdf_path = "/root/MMRAG/05-03-18-political-release.pdf"
#     output_folder = "test_output"

#         # 初始化数据库连接
#     text_db = QdrantManager(DB_PATH1, "text_collection")
#     image_db = QdrantManager(DB_PATH, "image_collection")
    
#     # 初始化embedder
#     text_embedder = ColBertEmbedder()
#     image_embedder = ColQwen2Embedder()

    
#     try:
#         # 1. 测试文档创建
#         print("\n1. 测试文档创建...")
#         doc = Document(pdf_path, text_db,image_db,text_embedder,image_embedder,output_folder)
#         print(f"文档创建成功，ID: {doc.doc_id}")
#         print(f"文档输出文件夹: {doc.output_folder}")
        
#         # 2. 测试文档处理
#         print("\n2. 测试文档处理...")
#         success = doc.document_process()
#         if success:
#             print("文档处理成功！")
#             print(f"提取的文本块数量: {len(doc.text_blocks)}")
#             print(f"处理的图片数量: {len(doc.images)}")

#             #3. 测试文档检索
#             qa_agent = QAAgent()

#             msg=[("user", "Among the adults conducted the survey on April 25 - May 1 2018, how many adults rated Trump's government ethical standards as poor? "),
#                  ]
            
#             result = qa_agent.run(msg,doc)
#             print(result)

#         else:
#             print("文档处理失败！")
        
#         # 4. 测试文档删除
#         print("\n4. 测试文档删除...")
#         if doc.delete():
#             print("文档删除成功！")
#             # 验证文件夹是否被删除
#             if not os.path.exists(doc.output_folder):
#                 print("输出文件夹已被清理")
#         else:
#             print("文档删除失败！")
            
#     except Exception as e:
#         print(f"测试过程中出错: {str(e)}")
#     finally:
#         # 清理测试文件夹
#         if os.path.exists(output_folder):
#             shutil.rmtree(output_folder)
#             print(f"\n清理测试文件夹: {output_folder}")

#         # 关闭数据库连接
#         try:
#             print("\n关闭数据库连接...")
#             text_db.close()
#             image_db.close()
#             print("数据库连接关闭成功！")
#         except Exception as e:
#             print(f"关闭数据库连接时出错: {str(e)}")
        