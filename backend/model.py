from abc import ABC, abstractmethod

class Model(ABC):  
    @abstractmethod
    def __init__(self, model_name: str):
        self.model_name = model_name

    @abstractmethod
    def construct_message_frame(self, role="user", content=""):
        pass

    @abstractmethod
    def process_message(self, message, model, history=[]):
        pass
