import os
import requests
import numpy as np
from prompt.llm_context_prompt import generate_llm_prompt
from dotenv import load_dotenv

load_dotenv()
# LLM_MODEL_PROVIDER = os.getenv("LLM_MODEL_PROVIDER")
LLM_MODEL_NAME = os.getenv("LLM_MODEL_NAME")
LLM_MODEL_API_URL = os.getenv("LLM_MODEL_API_URL")
# LLM_MODEL_API_KEY = os.getenv("LLM_MODEL_API_KEY")


class PlatServedModels:
    """
    class that implements two methods to be called from plat served model
    """

    def __init__(self, model_name: str = LLM_MODEL_NAME):
        self.model_name = LLM_MODEL_NAME
        self.llm_model_api_url = LLM_MODEL_API_URL
        self.api_endpoint = self.llm_model_api_url + "/api/generate"

    def generate_response(self, context, question):
        prompt = generate_llm_prompt(context, question)
        response = requests.post(
            self.api_endpoint,
            json={"model": self.model_name, "prompt": prompt, "stream": False},
        )
        return response.json()["response"]
