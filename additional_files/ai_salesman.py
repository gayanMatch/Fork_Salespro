from helpers.sales_gpt import SalesGPT
from langchain.chat_models import ChatOpenAI

class AISalesman:
    def __init__(self):
        self.llm = ChatOpenAI(temperature=0.9)
        self.sales_agent = SalesGPT.from_llm(self.llm)

    def get_response(self, user_input):
        self.sales_agent.human_step(user_input)
        agent_response = self.sales_agent.conversation_history[-1].replace('<END_OF_TURN>', '')
        return agent_response
