import agentscope
from configs import config

model_config = {
    "config_name": "base", # A unique name for the model config.
    "model_type": "dashscope_chat",    # Choose from "openai_chat", "openai_dall_e", or "openai_embedding".

    "model_name": "qwen-max",   # The model identifier used in the OpenAI API, such as "gpt-3.5-turbo", "gpt-4", or "text-embedding-ada-002".
    "api_key": config['key']['bailian_api_key'],               # Your OpenAI API key. If unset, the environment variable OPENAI_API_KEY is used.
    "organization": "",          # Your OpenAI organization ID. If unset, the environment variable OPENAI_ORGANIZATION is used.
}

agentscope.init(model_configs=[model_config])