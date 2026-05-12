import os
from langchain_openai import ChatOpenAI

class LLMProvider:
    @staticmethod
    def get_nvidia_llm(model_name="meta/llama-3.3-70b-instruct", temperature=0.2):
        """
        Returns a LangChain-compatible ChatOpenAI object configured for NVIDIA.
        """
        api_key = os.environ.get("nvidia_api_key")
        if not api_key:
            raise ValueError("NVIDIA_API_KEY not found in environment variables.")

        return ChatOpenAI(
            model=model_name,
            openai_api_key=api_key,
            openai_api_base="https://integrate.api.nvidia.com/v1",
            temperature=temperature,
            max_tokens=1024
        )