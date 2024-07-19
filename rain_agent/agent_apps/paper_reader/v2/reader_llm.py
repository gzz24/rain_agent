from langchain_openai import ChatOpenAI
# from rain_agent.configs import config

config = {
    'key': {
        'bailian_api_key': 'sk-9c3717935de94e339b10435852fe760e',
        'bailian_base_url': 'https://dashscope.aliyuncs.com/compatible-mode/v1'}}

# llm_core负责核心逻辑，llm_long负责对长文本处理
llm_core = ChatOpenAI(
    model='qwen-max',
    openai_api_key=config['key']['bailian_api_key'],
    openai_api_base=config['key']['bailian_base_url']
)

llm_long = ChatOpenAI(
    model='qwen-long',
    openai_api_key=config['key']['bailian_api_key'],
    openai_api_base=config['key']['bailian_base_url'],
)

