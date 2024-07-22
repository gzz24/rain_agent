"""
将一篇 paper 表示为函数调用
"""
import json
from typing import TypedDict, Annotated, List, Dict
from operator import add

from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langgraph.graph import add_messages, START, END, StateGraph

from rain_agent.configs import config


class FunctionState(TypedDict):
    functions: List[str]



# Models
llm_core = ChatOpenAI(
    model='qwen-max',
    openai_api_key=config['key']['bailian_api_key'],
    openai_api_base=config['key']['bailian_base_url']
)

