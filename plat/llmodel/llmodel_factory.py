from plat.llmodel.llmodels_microplat import PlatServedModels
from plat.llmodel.llmodels_cloud import (
    LLM,
    GPTModel,
    OllamaModel,
    AnthropicModel,
)


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
            return GPTModel(model_name=self.llmodel_name, api_key=None)
        elif self.llmodel_provider == "claude":
            return AnthropicModel(model_name=self.llmodel_name, api_key=None)
        else:
            raise ValueError(f"Unsupported model type: {self.llmodel_provider}")
