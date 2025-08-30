from backend.model import Model
from dashscope import Generation
from http import HTTPStatus
import random
from backend import config

class Deepseek(Model):
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.model_description = config.models_description[model_name]
        self.model_type = config.models_type[model_name]

    def construct_message_frame(self, role="user", content=""):
        """创建一个包含角色和内容的消息字典。"""
        return {'role': role, 'content': content}
    
    def process_message(self, message, model, history, enable_thinking):
        msg = self.construct_message_frame(content=message)
        history = history.copy()
        history.append(msg)
        
        if enable_thinking != True:
            response = Generation.call(model=model,
                                    messages=history,
                                    seed=random.randint(1, 10000),
                                    extra_body={"enable_thinking": enable_thinking})

            ret_text = ""
            if response.status_code == HTTPStatus.OK:
                ret_text = response.output["choices"][0]["message"]["content"]
            else:
                print('Request id: %s, Status code: %s' % (
                    response.request_id,
                    response.status_code))
                return "Error: 请求失败"
            
            return ret_text

        else:

            completion = Generation.call(
                model="deepseek-v3.1",
                messages=history,
                seed=random.randint(1, 10000),
                enable_thinking=True,
                result_format="message",
                # extra_body={"enable_thinking": enable_thinking},
                stream=True,
                incremental_output=True,
            )

            reasoning_content = ""      # 定义完整思考过程
            answer_content = ""         # 定义完整回复
            is_answering = False        # 判断是否结束思考过程并开始回复

            title_think = "=" * 20 + "思考过程" + "=" * 20 + "\n"
            title_ans = "\n" + "=" * 20 + "完整回复" + "=" * 20 + "\n"

            reasoning_content += title_think
            answer_content += title_ans
            for chunk in completion:
                if (
                    chunk.output.choices[0].message.content == ""
                    and chunk.output.choices[0].message.reasoning_content == ""
                ):
                    pass
                else:
                    # 如果当前为思考过程
                    if (
                        chunk.output.choices[0].message.reasoning_content != ""
                        and chunk.output.choices[0].message.content == ""
                    ):
                        reasoning_content += chunk.output.choices[0].message.reasoning_content

                    # 如果当前为回复
                    elif chunk.output.choices[0].message.content != "":
                        if not is_answering:
                            is_answering = True

                        answer_content += chunk.output.choices[0].message.content

            return reasoning_content + answer_content
