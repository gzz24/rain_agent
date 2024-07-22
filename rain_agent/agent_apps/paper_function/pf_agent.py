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


class FunctionState(TypedDict):
    functions: List[str]