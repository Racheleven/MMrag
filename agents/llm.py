
from openai import OpenAI
from langsmith import traceable
from langsmith.wrappers import wrap_openai

class myVLM:
    def __init__(self, model_name="Qwen/Qwen2.5-VL-32B-Instruct"):
        self.client = OpenAI(
            api_key="",
            base_url="https://api.siliconflow.cn/v1"
        )
        # self.model_name="Pro/Qwen/Qwen2.5-VL-7B-Instruct"
        self.model_name="Qwen/Qwen2.5-VL-32B-Instruct"
    

    def create_messages(self,image_path,system_prompt):
        import base64

        def encode_image(image_path):
            with open(image_path, "rb") as image_file:
                return base64.b64encode(image_file.read()).decode("utf-8")
        # # Path to your image
        # image_path = "/root/autodl-tmp/test_images/05-03-18-political-release.pdf_page_1.png"

        # Getting the Base64 string
        base64_image = encode_image(image_path)

        msg=[
                {
                    "role": "system",
                    "content": [{"type":"text","text":system_prompt}]},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "图片资料如下所示："},

                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{base64_image}"}, 
                        },
                        
                    ],
                }
            ]
        return msg
    
    @traceable(run_type="llm", name="qwen2.5-vl")
    def invoke(self,image_path,system_prompt):
        msg=self.create_messages(image_path,system_prompt)

        completion = self.client.chat.completions.create(
            model=self.model_name,
            messages=msg,
        )

        return completion
    