from flask import Flask, request, jsonify, render_template
import random
import dashscope
import os
import json
from backend import config
from backend.model_lib import qwen3, deepseekv3, farui_plus, tongyi_intent, qwen_math
from backend.multi_model_ensemble.router import Router
from backend.multi_model_ensemble.aggregator import Aggregator
import time
dashscope.api_key = open("./key.txt", 'r', encoding='utf-8').readline()

app = Flask(__name__)

# 加载历史对话
if os.path.exists('conversations.json'):
    with open('conversations.json', 'r', encoding='utf-8') as f:
        conversations_data = json.load(f)
else:
    conversations_data = {"dialogs": {}, "current_dialog_id": None}

# 定义可选择的模型列表
available_models_name = config.available_models
# 初始化全局变量
current_model = available_models_name[0] # qwen3-32b
# 实例化模型库
available_models = {
    "qwen3-32b": qwen3.QWen3("qwen3-32b"),
    "qwen3-0.6b": qwen3.QWen3("qwen3-0.6b"),
    "deepseek-v3": deepseekv3.Deepseek("deepseek-v3"),
    "qwen-math-turbo": qwen_math.QWenMath("qwen-math-turbo"),
    "qwen3-coder-flash": qwen3.QWen3("qwen3-coder-flash"),
    "tongyi-intent-detect-v3": tongyi_intent.TongyiIntent("tongyi-intent-detect-v3"),
    "farui-plus": farui_plus.Farui("farui-plus")
}

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html', dialogs=conversations_data["dialogs"], current_dialog_id=conversations_data["current_dialog_id"], models=available_models_name, current_model=current_model)


@app.route('/send_message', methods=['POST'])
def send_message():
    global current_model
    try:
        data = request.form
        message = data.get('message')
        enable_thinking = request.form.get('deep_thinking', 'true').lower() == 'true'
        current_dialog_id = conversations_data.get("current_dialog_id")
        if not current_dialog_id:
            return jsonify({'error': '请选择或创建一个对话。'})
        
        if current_dialog_id not in conversations_data["dialogs"]:
            conversations_data["dialogs"][current_dialog_id] = {
                "name": f"对话{current_dialog_id}",
                "conversations": []
            }

        dialog = conversations_data["dialogs"][current_dialog_id]
        history = dialog["conversations"]
        # 使用单模型回答策略
        if current_model != "ensemble-chat(ours)":
            # 建立模型
            model = available_models[current_model]
            start_time = time.time()  # 记录开始时间
            # 处理消息
            response = model.process_message(message, current_model, history, enable_thinking)
            end_time = time.time()  # 记录结束时间
            response = f"(use {current_model}, consume: {end_time - start_time}s)\n" + response
            dialog["conversations"].append({"role": "user", "content": message})
            dialog["conversations"].append({"role": "system", "content": response})

        else:
            # 先进行意图识别
            model = available_models["tongyi-intent-detect-v3"]
            start_time = time.time()  # 记录开始时间
            purpose = model.process_message(message, "tongyi-intent-detect-v3", history=[], enable_thinking=False)
            # 基于意图识别进行路由选择，集成得到多智能体集群
            router = Router()
            multi_ensembled_agent = router.process_message(message, "tongyi-intent-detect-v3", purpose)
            print(multi_ensembled_agent)
            multi_ensembled_agent_lst = [part.strip() for part in multi_ensembled_agent.split(',')] 
            # 多智能体集群构成的“超网”实行聚合策略(这里使用debate)，输出优化的结果
            aggregator = Aggregator("debate")
            message_purpose = message + f" after analysing query text, we summary the query\'s purpose is {purpose}"
            response = aggregator.aggregate(available_models, multi_ensembled_agent_lst, message_purpose, history, enable_thinking)
            end_time = time.time()  # 记录结束时间
            response = f"(use ensemble-chat(ours) - ensembles [{multi_ensembled_agent}], consume: {end_time - start_time}s) \n" + response
            dialog["conversations"].append({"role": "user", "content": message})
            dialog["conversations"].append({"role": "system", "content": response})

        with open('conversations.json', 'w', encoding='utf-8') as f:
            json.dump(conversations_data, f, ensure_ascii=False, indent=4)
            
        return jsonify({'message': message, 'response': response})
       
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'服务器错误: {str(e)}'})

@app.route('/create_dialog', methods=['POST'])
def create_dialog():
    new_id = str(random.randint(10000, 99999))
    conversations_data["dialogs"][new_id] = {"name": f"对话{new_id}", "conversations": []}
    conversations_data["current_dialog_id"] = new_id

    with open('conversations.json', 'w', encoding='utf-8') as f:
        json.dump(conversations_data, f, ensure_ascii=False, indent=4)

    return jsonify({
        "dialogs": conversations_data["dialogs"], 
        "current_dialog_id": new_id,
        "conversations": []  # 添加空对话列表
    })

@app.route('/select_dialog', methods=['POST'])
def select_dialog():
    data = request.form
    dialog_id = data.get('dialog_id')
    if dialog_id in conversations_data["dialogs"]:
        conversations_data["current_dialog_id"] = dialog_id

        with open('conversations.json', 'w', encoding='utf-8') as f:
            json.dump(conversations_data, f, ensure_ascii=False, indent=4)

        return jsonify({
            "dialogs": conversations_data["dialogs"],  # 添加完整的dialogs数据
            "current_dialog_id": dialog_id, 
            "conversations": conversations_data["dialogs"][dialog_id]["conversations"]
        })
    else:
        return jsonify({'error': '对话ID不存在。'})


@app.route('/delete_dialog', methods=['POST'])
def delete_dialog():
    data = request.form
    dialog_id = data.get('dialog_id')

    if dialog_id in conversations_data["dialogs"]:
        del conversations_data["dialogs"][dialog_id]

        if conversations_data["current_dialog_id"] == dialog_id:
            conversations_data["current_dialog_id"] = None

        with open('conversations.json', 'w', encoding='utf-8') as f:
            json.dump(conversations_data, f, ensure_ascii=False, indent=4)

        return jsonify({
            "dialogs": conversations_data["dialogs"], 
            "current_dialog_id": conversations_data["current_dialog_id"],
            "conversations": []  # 添加空对话列表
        })
    else:
        return jsonify({'error': '对话ID不存在。'})

@app.route('/rename_dialog', methods=['POST'])
def rename_dialog():
    data = request.form
    dialog_id = data.get('dialog_id')
    new_name = data.get('new_name')

    if dialog_id in conversations_data["dialogs"]:
        conversations_data["dialogs"][dialog_id]["name"] = new_name

        with open('conversations.json', 'w', encoding='utf-8') as f:
            json.dump(conversations_data, f, ensure_ascii=False, indent=4)

        return jsonify({"dialogs": conversations_data["dialogs"], "current_dialog_id": conversations_data["current_dialog_id"]})
    else:
        return jsonify({'error': '对话ID不存在。'})


@app.route('/select_model', methods=['POST'])
def select_model():
    global current_model
    data = request.form
    model = data.get('model')
    if model in available_models_name:
        current_model = model
        return jsonify({'success': True, 'current_model': current_model})
    else:
        return jsonify({'error': '模型不存在。'})

@app.route('/init')
def init_app():
    """初始化应用，返回所有必要数据"""
    return jsonify({
        "dialogs": conversations_data["dialogs"],
        "current_dialog_id": conversations_data["current_dialog_id"],
        "models": available_models_name,
        "current_model": current_model
    })

if __name__ == '__main__':
    app.run(debug=True)