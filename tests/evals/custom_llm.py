import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from deepeval.models.base_model import DeepEvalBaseLLM

load_dotenv()

class CustomGroqLLM(DeepEvalBaseLLM):
    def __init__(self):
        # We explicitly use llama-3.3-70b-versatile for evaluation because it has excellent 
        # tool calling and structured output support which DeepEval relies on heavily.
        self.chat_model = ChatGroq(
            api_key=os.getenv("GROQ_API_KEY"),
            model="llama-3.3-70b-versatile" 
        )

    def load_model(self):
        return self.chat_model

    def generate(self, prompt: str, schema=None, **kwargs):
        if schema:
            model_with_structure = self.chat_model.with_structured_output(schema)
            return model_with_structure.invoke(prompt)
        res = self.chat_model.invoke(prompt)
        return res.content

    async def a_generate(self, prompt: str, schema=None, **kwargs):
        if schema:
            model_with_structure = self.chat_model.with_structured_output(schema)
            return await model_with_structure.ainvoke(prompt)
        res = await self.chat_model.ainvoke(prompt)
        return res.content

    def get_model_name(self):
        return "llama-3.3-70b-versatile"
