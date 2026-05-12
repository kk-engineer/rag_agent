from openai import OpenAI, APIConnectionError
from rich.console import Console

console = Console()
BASE_URL="http://localhost:8000/v1"
MISTRAL_MODEL="Mistral-Small-3bit-MLX"


def empty_stream():
    yield from []


client = OpenAI(
    base_url=BASE_URL,
    api_key="not-needed"
)


def chat_stream(
    messages,
    temperature=0.1
):

    try:
        return client.chat.completions.create(
            model=MISTRAL_MODEL,
            messages=messages,
            temperature=temperature,
            stream=True
        )

    except APIConnectionError:

        error_msg = (
            "LLM server connection failed.\n"
            "Possible reasons:\n"
            "1. Local LLM server is not running\n"
            "2. Wrong base_url/port\n"
            "3. Model server crashed\n"
            "4. Network/socket issue"
        )
        print(error_msg)
        return empty_stream()

    except Exception as e:
        error_msg = (
            f"LLM request failed: {str(e)}"
        )
        print(error_msg)

        return empty_stream()