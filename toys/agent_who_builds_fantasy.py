import os
import glob

config = {
    'key': {
        'bailian_api_key': 'sk-9c3717935de94e339b10435852fe760e',
        'bailian_base_url': 'https://dashscope.aliyuncs.com/compatible-mode/v1'
    }
}

from langgraph.graph import StateGraph
from langgraph.graph.message import add_messages
from typing import TypedDict, Annotated
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from operator import add


class State(TypedDict):
    messages: Annotated[list, add_messages]
    # 这个世界的运行规则
    world_rules: Annotated[list, add]


llm_core = ChatOpenAI(
    model='qwen-max',
    openai_api_key=config['key']['bailian_api_key'],
    openai_base_url=config['key']['bailian_base_url']
)


@tool
def validate_a_rule(rule: str):
    """validate whether the rule aligns with existing rules"""
