from prompt.llm_context_prompt import generate_llm_prompt
from abc import ABC, abstractmethod
from langchain_community.llms.ollama import Ollama
from openai import OpenAI
import anthropic

# from dotenv import load_dotenv

# load_dotenv()
# LLM_MODEL_PROVIDER = os.getenv("LLM_MODEL_PROVIDER")
# LLM_MODEL_NAME = os.getenv("LLM_MODEL_NAME")
# LLM_MODEL_API_URL = os.getenv("LLM_MODEL_API_URL")
# LLM_MODEL_API_KEY = os.getenv("LLM_MODEL_API_KEY")


class LLM(ABC):
    def __init__(self, model_name: str):
        self.model_name = model_name

    @abstractmethod
    def invoke(self, prompt: str) -> str:
        pass

    def generate_response(self, context: str, question: str) -> str:
        prompt = generate_llm_prompt(context, question)
        response_text = self.invoke(prompt)
        return response_text


class OllamaModel(LLM):
    def __init__(self, model_name: str):
        super().__init__(model_name)
        self.model = Ollama(model=model_name)

    def invoke(self, prompt: str) -> str:
        return self.model.invoke(prompt)


class GPTModel(LLM):
    def __init__(self, model_name: str, api_key: str):
        super().__init__(model_name)
        self.client = OpenAI(api_key=api_key)

    def invoke(self, prompt: str) -> str:
        messages = [
            # {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ]
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            max_tokens=150,
            n=1,
            stop=None,
            temperature=0.7,
        )
        return response.choices[0].message.content.strip()


class AnthropicModel(LLM):
    def __init__(self, model_name: str, api_key: str):
        super().__init__(model_name)
        self.client = anthropic.Anthropic(api_key=api_key)

    def invoke(self, prompt: str) -> str:
        messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
        response = self.client.messages.create(
            model=self.model_name, max_tokens=1000, temperature=0.7, messages=messages
        )
        # Extract the plain text from the response content
        text_blocks = response.content
        plain_text = "\n".join(
            block.text for block in text_blocks if block.type == "text"
        )
        return plain_text
