import random
from dashscope import Generation

class Debate_Two():
    def __init__(self, available_models, model1_name, model2_name):
        self.model1 = available_models[model1_name]
        self.model2 = available_models[model2_name]
        self.model1_name = model1_name
        self.model2_name = model2_name

    def execute(self, message, history=[], iter=2):
        history = history.copy()
        for i in range(iter):
            enable_thinking = random.choice([True, False])      # 0.5概率随机进行推理增强
            result1 = self.model1.process_message(message, self.model1_name, history, enable_thinking)
            enable_thinking = random.choice([True, False])      # 0.5概率随机进行推理增强
            result2 = self.model2.process_message(message, self.model2_name, history, enable_thinking)  
            history.append({"role": "system", "content": result1})
            history.append({"role": "system", "content": result2})
            print(f"finish {i}th iter! \n")

        # 最后再由专有模型进行决断，如果没有专有或者都是，就选第一个进行决断   
        if self.model1.model_type == "specific" and self.model2.model_type != "specific":
            enable_thinking = random.choice([True, False])      # 0.5概率随机进行推理增强
            final_result = self.model1.process_message(message, self.model1_name, history, enable_thinking)
        elif self.model1.model_type != "specific" and self.model2.model_type == "specific":
            enable_thinking = random.choice([True, False])      # 0.5概率随机进行推理增强
            final_result = self.model2.process_message(message, self.model2_name, history, enable_thinking)
        else:
            enable_thinking = random.choice([True, False])      # 0.5概率随机进行推理增强
            final_result = self.model1.process_message(message, self.model1_name, history, enable_thinking)

        return final_result
    
class Debate_Three():
    def __init__(self, available_models, model1_name, model2_name, model3_name):
        self.model1 = available_models[model1_name]
        self.model2 = available_models[model2_name]
        self.model3 = available_models[model3_name]
        self.model1_name = model1_name
        self.model2_name = model2_name
        self.model3_name = model3_name

    def judge_model(self, judge_model, query, message):
        role_set = [{'role': 'system', 'content': f'You are a judge to evaluate the user\'s reply quality. \
                     The query is {query}. Review the query and check the user\'s answer, directly give it the score(int range from 0~100)'}]
        judge_set = [{'role': 'user', 'content': message}]
        one_judge = role_set + judge_set
        response = Generation.call(model=judge_model.model_name,
                                messages=one_judge,
                                extra_body={"enable_thinking": False})
        score = response.output["choices"][0]["message"]["content"]
        print(score)

        return int(score)

    def execute(self, message, history=[], iter=2):
        history = history.copy()
        # 选择其中任意一个general来当做“裁判员”，如果没有就选择其中第一个即可
        if self.model1.model_type == "general":
            judge_model = self.model1
            debate_model1 = self.model2
            debate_model2 = self.model3
        elif self.model2.model_type == "general":
            judge_model = self.model2
            debate_model1 = self.model1
            debate_model2 = self.model3
        elif self.model3.model_type == "general":
            judge_model = self.model3
            debate_model1 = self.model1
            debate_model2 = self.model2
        else:
            judge_model = self.model1
            debate_model1 = self.model2
            debate_model2 = self.model3

        score = []
        result = []
        for i in range(iter):
            enable_thinking = random.choice([True, False])      # 0.5概率随机进行推理增强
            result1 = debate_model1.process_message(message, debate_model1.model_name, history, enable_thinking)
            enable_thinking = random.choice([True, False])      # 0.5概率随机进行推理增强
            result2 = debate_model2.process_message(message, debate_model2.model_name, history, enable_thinking)  
            history.append({"role": "user", "content": result1})
            history.append({"role": "user", "content": result2})

            result.append({"role": "system", "content": result1})
            result.append({"role": "system", "content": result2})
            score1 = self.judge_model(judge_model, message, result1)
            score2 = self.judge_model(judge_model, message, result2)
            score.append(score1)
            score.append(score2)

            print(f"finish {i}th iter! \n")

        # 最后选择所有结果里头分数最高的一个作为最终答案
        max_index = max(range(len(score)), key=lambda i: score[i])
        final_result = result[max_index]

        return final_result