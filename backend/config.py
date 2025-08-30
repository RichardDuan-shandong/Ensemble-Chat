# 定义模型的列表
available_models = ["qwen3-0.6b",                            # qwen light model
                    "deepseek-v3",                           # deepseek v3
                    "qwen3-32b",                             # qwen-base model
                    "qwen-math-turbo",                       # qwen model for math
                    "qwen3-coder-flash",                     # qwen model for code
                    "farui-plus",                            # farui model for laws task
                    "tongyi-intent-detect-v3",               # intent detect
                    "ensemble-chat(ours)"                    # our ensemble strategy
                    ] 

# 模型描述列表
models_description = {
    "qwen3-0.6b": "qwen千问0.6B轻量级模型，适合解决简单的任务，运行轻便快速，但是推理能力较弱。",
    "deepseek-v3": "DeepSeek-V3 为MoE 模型，在长文本、代码、数学、百科、中文能力上表现优秀，适用于复杂的任务与较多知识检索与利用的场景。",
    "qwen3-32b": "qwen千问32B开源基座模型，具备推理与非推理结合功能，深度思考能力强，适合复杂决策判断、逻辑推导等日常具有一定挑战性的任务",
    "qwen-math-turbo" : "通义千问数学模型是专门用于数学解题的语言模型，致力于解决复杂、具有挑战性的数学问题",
    "qwen3-coder-flash" : "Qwen3-Coder 模型具有强大的代码能力，并且运行高效快速。",
    "farui-plus": "通义法睿是以通义千问为基座经法律行业数据和知识专门训练的法律行业大模型产品，具有回答法律问题、推理法律适用、推荐裁判类案、辅助案情分析、生成法律文书、检索法律知识、审查合同条款等功能。",
    "tongyi-intent-detect-v3" : "通义千问的意图理解模型能够在百毫秒级时间内快速、准确地解析用户意图，并选择合适的工具来解决用户的问题。",
    "ensemble-chat(ours)": "各类大小混合模型的智能集成，给予最佳的回答！"
}

# 模型属性列表
models_type = {
    "qwen3-0.6b": "light",
    "deepseek-v3": "general",
    "qwen3-32b": "general",
    "qwen-math-turbo" : "specific",
    "qwen3-coder-flash" : "specific",
    "farui-plus": "specific",
    "tongyi-intent-detect-v3" : "specific",
    "ensemble-chat(ours)": "ensemble"
}
