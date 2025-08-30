from backend.model import Model
from dashscope import Generation
from http import HTTPStatus
import random
from backend import config
class QWenMath(Model):
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.model_description = config.models_description[model_name]
        self.model_type = config.models_type[model_name]
        
    def construct_message_frame(self, role="user", content=""):
        """创建一个包含角色和内容的消息字典。"""
        return {'role': role, 'content': content}
    
    def process_message(self, message, model, history, enable_thinking):
        msg = self.construct_message_frame(content=message)
        response = Generation.call(model=model,
                                messages=[msg],
                                result_format='message')

        if response.status_code == HTTPStatus.OK:
            ret_text = response.output["choices"][0]["message"]["content"]
        else:
            print('Request id: %s, Status code: %s' % (
                response.request_id,
                response.status_code))
            return "Error: 请求失败"
        
        return ret_text
    
      