from backend.multi_model_ensemble.tools import debate

class Aggregator():
    def __init__(self, aggregate_strategy="debate"):
        self.aggregate_strategy = aggregate_strategy    # there is only "debate" tool now
            
    def aggregate(self, available_models, multi_agent_lst, message, history=[], enable_thinking=False):
        history = history.copy()
        if len(multi_agent_lst) == 1:
            model = available_models[multi_agent_lst[0]]
            response = model.process_message(message, multi_agent_lst[0], history, enable_thinking)
        elif len(multi_agent_lst) == 2:
            if self.aggregate_strategy == "debate":
                aggregator = debate.Debate_Two(available_models, multi_agent_lst[0], multi_agent_lst[1])
                response = aggregator.execute(message, history, iter=2)
        else:           # max is 3
            if self.aggregate_strategy == "debate":
                aggregator = debate.Debate_Three(available_models, multi_agent_lst[0], multi_agent_lst[1], multi_agent_lst[2])
                response = aggregator.execute(message, history, iter=2)
            
        return response