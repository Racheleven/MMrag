import gradio as gr
import random
import time
import os
import shutil
from agents.document import Document
from agents.qa_agent import QAAgent
from models.embedder import ColQwen2Embedder, ColBertEmbedder
from models.database import QdrantManager
from config.settings import PROCESS_IMAGE_OUTPUT_FOLDER, DB_PATH, DB_PATH1, DEFAULT_BATCH_SIZE


# 初始化数据库连接和embedder（全局只需要一次）
text_db = QdrantManager(DB_PATH1, "text_collection")
image_db = QdrantManager(DB_PATH, "image_collection")
import subprocess
import os

# 保存原始的环境变量
original_env = {var: os.environ.get(var) for var in ['http_proxy', 'https_proxy', 'no_proxy']}

# 改变环境变量
result = subprocess.run('bash -c "source /etc/network_turbo && env | grep proxy"', shell=True, capture_output=True, text=True)
output = result.stdout
for line in output.splitlines():
    if '=' in line:
        var, value = line.split('=', 1)
        os.environ[var] = value
        print(var, value)


text_embedder = ColBertEmbedder()
image_embedder = ColQwen2Embedder()



# 后面需要恢复的时候
for var, value in original_env.items():
    if value is None:
        os.environ.pop(var, None)  # 原本没有，就删掉
    else:
        os.environ[var] = value     # 原本有，就还原成原来的值有，就还原成原来的值
        
        
qa_agent = QAAgent()

# 定义检索模式
SEARCH_MODES = {
    "文本块检索模式": "text",
    "双检索模式": "all",
    "文档截图检索模式": "image"
}

# 定义图片处理模式
IMAGE_PROCESS_MODES = {
    "单页处理": "single",
    "相邻页合并": "merge"
}

# 全局变量存储当前文档和检索模式
current_doc = None
current_mode = "all"  
current_image_mode = "single"  # 默认使用single模式
uploaded_folder = "uploaded_docs"
os.makedirs(uploaded_folder, exist_ok=True)

def process_pdf(file, history):
    """处理上传的PDF文件"""
    global current_doc
    
    try:
        if not file:
            return "请选择要上传的PDF文件", history
            
        # 保存上传的文件
        filename = os.path.basename(file.name)
        save_path = os.path.join(uploaded_folder, filename)
        shutil.copy2(file.name, save_path)
        
        # 创建新的Document对象
        doc = Document(
            save_path, 
            text_db,
            image_db,
            text_embedder,
            image_embedder,
            output_folder=PROCESS_IMAGE_OUTPUT_FOLDER,
            image_process_mode=current_image_mode  # 使用当前选择的图片处理模式
        )
        
        # 处理文档
        success = doc.document_process()
        if not success:
            return f"文档处理失败: {filename}", history
            
        # 更新当前文档
        if current_doc:
            try:
                current_doc.delete()  # 删除旧文档
            except:
                pass
                
        current_doc = doc
        
        # 清空聊天历史
        return f"文档上传并处理成功！\nID: {doc.doc_id}\n文本块数量: {len(doc.text_blocks)}\n图片数量: {len(doc.images)}", []
        
    except Exception as e:
        return f"文档处理失败: {str(e)}", history

def delete_current_pdf():
    """删除当前PDF文档"""
    global current_doc
    
    if not current_doc:
        return None, None, None
        
    try:
        # 删除文档数据
        current_doc.delete()
        
        # 删除文件
        if os.path.exists(current_doc.pdf_path):
            os.remove(current_doc.pdf_path)
            
        current_doc = None
        # 返回None来清空file_upload和upload_output，以及chatbot历史
        return None, None, []
        
    except Exception as e:
        return None, None, None

def bot(history: list):
    print(history)
    if not current_doc:
        history.append({"role": "assistant", "content": "请先上传PDF文档"})
        yield history
        return
            
    print(history)
    # 根据当前模式进行检索
    bot_message = qa_agent.run(history, current_doc, mode=current_mode)
    history.append({"role": "assistant", "content": bot_message})
    yield history

def change_mode(mode):
    """更改检索模式"""
    global current_mode
    current_mode = SEARCH_MODES[mode]
    print(current_mode)

def change_image_mode(mode):
    """更改图片处理模式"""
    global current_image_mode
    current_image_mode = IMAGE_PROCESS_MODES[mode]
    print(current_image_mode)

with gr.Blocks(title="智能PDF问答系统",theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🔍智能PDF问答系统")
    with gr.Row():
        with gr.Column(scale=1):
            # 添加检索模式选择
            mode_select = gr.Radio(
                choices=list(SEARCH_MODES.keys()),
                value="双检索模式",
                label="选择检索模式"
            )
            
            # 添加图片处理模式选择
            image_mode_select = gr.Radio(
                choices=list(IMAGE_PROCESS_MODES.keys()),
                value="相邻页合并",
                label="选择图片处理模式"
            )
            
            # PDF管理部分
            file_upload = gr.File(label="选择PDF文件")
            upload_button = gr.Button("上传并处理PDF")
            upload_output = gr.Textbox(label="上传结果")
            delete_button = gr.Button("删除当前PDF")
            
        with gr.Column(scale=2):
            # 聊天部分
            chatbot = gr.Chatbot(
                type="messages",
                height=490  # 设置聊天框的高度为600像素
            )
            msg = gr.Textbox(label="输入问题")
            clear = gr.Button("清空对话")

    def user(user_message, history: list):
        history.append({"role": "user", "content": user_message})
        return "", history

    # 处理PDF上传
    upload_button.click(
        process_pdf,
        inputs=[file_upload, chatbot],
        outputs=[upload_output, chatbot],
        show_progress_on=upload_output
    )
    
    # 处理PDF删除
    delete_button.click(
        delete_current_pdf,
        outputs=[file_upload, upload_output, chatbot],
    )
    
    # 处理模式切换
    mode_select.change(
        change_mode,
        inputs=[mode_select]
    )
    
    # 处理图片处理模式切换
    image_mode_select.change(
        change_image_mode,
        inputs=[image_mode_select]
    )
    
    # 处理聊天
    msg.submit(user, [msg, chatbot], [msg, chatbot], queue=False).then(
        bot, chatbot, chatbot
    )
    
    # 修改清空对话按钮的实现
    clear.click(lambda: [], None, chatbot, queue=False)

if __name__ == "__main__":
    demo.launch()