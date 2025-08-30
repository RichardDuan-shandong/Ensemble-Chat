from backend.model import Model
from dashscope import Generation
from http import HTTPStatus
from backend import config
import json

class TongyiIntent(Model):
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.model_description = config.models_description[model_name]
        self.model_type = config.models_type[model_name]
        self.tools = [
            {
                "name": "coding",
                "description": "用户意图执行编写代码的任务。",
                "parameters": {}
            },
            {
                "name": "normal",
                "description": "用户意图执行的任务未属于各特定领域工具，归类为通用常规对话意图，这类是比较普遍的",
                "parameters": {}
            },
            {
                "name": "laws_suggest",
                "description": "用户意图执行法律咨询、审查合同条款、生成法律文书、检索法律知识等法律相关的事务。",
                "parameters": {}
            },
            {
                "name": "math",
                "description": "用户意图解决一个较为复杂的数学计算、数学逻辑推理问题",
                "parameters": {}
            },

            {
                "name": "text_emotion_classification",
                "description": "在用户指明这是一个情感分类任务时，用户意图完成一个文本情感分类任务，针对用户给出的文本，给予相应的评价性词语并具有积极或消极等倾向",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "task": {
                            "type": "string",
                            "description": "task这时候必须为classification",
                        },
                        "sentence": {
                            "type": "string",
                            "description": "用户输入的需要处理的文本内容，支持中英文 例如“老师今天表扬我了”，“虽然包装有一些破损，但发货和运送很快”",
                        },
                        "labels": {
                            "type": "string",
                            "description": "label为分类体系，具体而言，在这个情感分类任务中代表了情感倾向或者情感感受词语，比如积极/中立/消极，或者更细致的情感感受：高兴、惊讶、愤怒、悲伤、惊喜、羡慕、嫉妒等等，可以有多个label，label有可能用户会暗示给出，各个label用英文半角逗号分隔开",
                        },
                    },
                    "required": ["task", "sentence", "labels"]
                    }
            },

            {
                "name": "text_topic_classification",
                "description": "在用户指明这是一个主题分类任务时，完成一个文本主题分类任务，针对用户给出的文本，判断其所述的主要内容或所属的主题领域，这种问题需要有明显的意图暗示，比如说就是让分析某句话或者某个新闻的类别，如果没有不应当分为此类",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "task": {
                            "type": "string",
                            "description": "任务类型，此时必须固定为'classification'",
                        },
                        "sentence": {
                            "type": "string",
                            "description": "用户输入的需要进行主题分类的文本内容，支持中英文。例如'央行宣布下调金融机构存款准备金率0.25个百分点'，'欧冠半决赛首回合，皇马在主场与拜仁战成2-2平'",
                        },
                        "labels": {
                            "type": "string",
                            "description": "label为分类体系，具体而言，在这个主题分类任务中代表了所有可能的话题或领域类别。例如财经/体育/科技/娱乐/健康/教育/汽车/房产等。用户可能会直接给出类别，也可能在对话中暗示。各个label用英文半角逗号分隔开。如果用户未明确指定，应询问或使用一个通用的主题分类体系。",
                        },
                    },
                    "required": ["task", "sentence", "labels"]
                }
            },

            {
                "name": "named_entity_recognition",
                "description": "用户意图完成一个命名实体识别任务，针对用户给出的文本，从中抽取出指定类型的实体对象",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "task": {
                            "type": "string",
                            "description": "任务类型，此时必须固定为'extraction'",
                        },
                        "sentence": {
                            "type": "string",
                            "description": "用户输入的需要从中抽取实体的文本内容，支持中英文。例如'马云于1964年9月10日生于浙江省杭州市'，'本次会议将于下周一下午在腾讯北京总部举行'",
                        },
                        "labels": {
                            "type": "string",
                            "description": "label为需要从文本中抽取的实体类型。例如人名/地名/组织机构名/时间/日期/金额/产品名等。用户会明确指定需要抽取的实体类型。各个label用英文半角逗号分隔开。",
                        },
                    },
                    "required": ["task", "sentence", "labels"]
                }
            },

            {
                "name": "machine_reading_comprehension",
                "description": "用户意图完成一个机器阅读理解任务，针对用户给出的文本和一系列问题，从文本中抽取出每个问题对应的答案",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "task": {
                            "type": "string",
                            "description": "任务类型，此时必须固定为'extraction'",
                        },
                        "sentence": {
                            "type": "string",
                            "description": "用户输入的包含答案的源文本内容，支持中英文。例如'小明今年15岁，他的身高是175厘米，就读于第一中学。'",
                        },
                        "labels": {
                            "type": "string",
                            "description": "label为需要基于文本回答的具体问题。多个问题用英文半角逗号分隔开。例如'小明的年龄是多少？,小明的身高是多少？,小明在哪里上学？'。用户会明确给出想要提问的问题。",
                        },
                    },
                    "required": ["task", "sentence", "labels"]
                }
            }

        ]

    def construct_message_frame(self, role="user", content=""):
        """创建一个包含角色和内容的消息字典。"""
        return {'role': role, 'content': content}
    
    def process_message(self, message, model, enable_thinking=False, history=[]):
        tools_string = json.dumps(self.tools,ensure_ascii=False)
        
        system_prompt = f""" You are a helpful assistant. You may call one or more tools to assist with the user query. The tools you can use are as follows:
        {tools_string}  Response in INTENT_MODE."""
        
        msg = self.construct_message_frame(content=message)
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