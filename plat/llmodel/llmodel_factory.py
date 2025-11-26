from plat.llmodel.llmodels_microplat import PlatServedModels
from plat.llmodel.llmodels_cloud import (
    LLM,
    GPTModel,
    OllamaModel,
    AnthropicModel,
)
from dotenv import load_dotenv
import os

load_dotenv()


class LLModelFactory:

    def __init__(self, llmodel_provider: str, model_name: str, api_key: str = None):
        self.llmodel_provider = llmodel_provider
        self.llmodel_name = model_name
        self.api_key = api_key

    def get_llmodel_accessor(self):
        if self.llmodel_provider == "local":
            return PlatServedModels(model_name=self.llmodel_name)
        elif self.llmodel_provider == "plat":
            return PlatServedModels(model_name=self.llmodel_name)
        elif self.llmodel_provider == "ollama":
            return OllamaModel(model_name=self.llmodel_name)
        elif self.llmodel_provider == "gpt":
            api_key = os.getenv("OPENAI_API_KEY") or self.api_key
            if not api_key:
                raise ValueError("OpenAI API key is required for GPT models")
            return GPTModel(model_name=self.llmodel_name, api_key=api_key)
        elif self.llmodel_provider == "claude":
            api_key = os.getenv("ANTHROPIC_API_KEY") or self.api_key
            if not api_key:
                raise ValueError("Anthropic API key is required for Claude models")
            return AnthropicModel(model_name=self.llmodel_name, api_key=api_key)
        else:
            raise ValueError(f"Unsupported model type: {self.llmodel_provider}")
