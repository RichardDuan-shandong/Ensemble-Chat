from backend.model import Model
from dashscope import Generation
from http import HTTPStatus
from backend import config
import json
from backend.config import available_models, models_description
class Router(Model):
    def __init__(self):
        self.model_name = "tongyi-intent-detect-v3"
        self.model_description = config.models_description["tongyi-intent-detect-v3"]
        # 获得可选的混合模型集群
        self.model_soup =  {model: models_description[model] for model in available_models[:6]}

    def construct_message_frame(self, role="user", content="", purpose=""):
        """创建一个包含角色和内容的消息字典。"""
        return {'role': role, 'content': "the user's primitive query is:" + content + " and we extract the user's purpose and query's feature, it is" + purpose}
    
    def process_message(self, message, model, purpose):
        model_soup_string = json.dumps(self.model_soup,ensure_ascii=False)

        system_prompt = f""" You are a helpful assistant. You should choose 1~3 proper model(s) from the ensemble multi-agent LLMs system to answer user's query based on their feature and \
                        advantages: {model_soup_string}. Our aim is to achieve the balance between inference time and performance, to make our system faster and has less hallucination . You should
                        think carefully on the belonging domain and the difficulty of user's query. Just reply with the chosen model(s)(format is just divided by , no other things). \
                        (note: except some really easy query , most of time using multi-agent is beneficial to lower hallucination. )"""
                        
        msg = self.construct_message_frame(content=message, purpose=purpose)
        messages = [{'role': 'system', 'content': system_prompt}] + [msg]

        response = Generation.call(model=model,
                                   messages=messages,
                                   result_format='message')

        if response.status_code == HTTPStatus.OK:
            return response.output["choices"][0]["message"]["content"]
        else:
            print('Request id: %s, Status code: %s, Detail: %s' % (
                response.request_id,
                response.status_code,
                response.detail if response.detail else ''))
            return "Error: 请求失败"