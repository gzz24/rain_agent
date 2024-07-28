# 固化代码
import json
from typing import List, Dict, Literal, Union

from configs import config
from prompts import *
from tools import *
from langchain.globals import set_verbose, set_debug

from langgraph.graph import START, END, StateGraph, add_messages
from typing import TypedDict, List, Dict, Annotated
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import ToolNode
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage, trim_messages
# end

import tiktoken


def str_token_counter(msgs: list) -> int:
    enc = tiktoken.get_encoding("o200k_base")
    cnt = 0
    for e in msgs:
        cnt += len(enc.encode(e.content))
    return cnt


trimmer = trim_messages(
    max_tokens=5000,
    strategy="last",
    token_counter=str_token_counter,
    include_system=True,
)


# 固化模板

# -- graph
class State(TypedDict):
    messages: Annotated[list, add_messages]
    opinion: str


# -- llm and tool
llm = ChatOpenAI(
    model='qwen-max',
    openai_api_key=config['key']['bailian_api_key'],
    openai_api_base=config['key']['bailian_base_url']
)


def call_model(state: State):
    messages = state['messages']
    response = ''
    print('model:', end='')
    for chunk in llm.stream(messages):
        res = chunk.content
        response += res
        print(res, end='', flush=True)
    print('\n')
    return {
        'messages': [response]
    }


def change_opinion(new_opinion: Literal['a = 1', 'a = 2', 'a = 3']):
    """根据对话判断是否需要改变自己的观点"""
    return new_opinion


def change_status(state: State):
    llm_with_tool = llm.bind_tools([change_opinion])

    last_messages = state['messages'][-1]
    opinion = state['opinion']

    res = llm_with_tool.invoke(
        [
            SystemMessage(content=f'请你根据对话判断，是否需要对你自己的观点进行更改。你的观点是: {opinion}'),
            HumanMessage(content=f"上一条对话是: \"{last_messages.content}\"")
        ]
    )

    if len(res.tool_calls) != 0:
        print(f"opinion changed to {res.tool_calls[0]['args']['new_opinion']}")
        return {
            'opinion': res.tool_calls[0]['args']['new_opinion']
        }
    else:
        return {
            'opinion': opinion
        }


def user_response(state: State):
    user_msg = input('user:')

    return {
        'messages': [HumanMessage(content=user_msg)]
    }


def route(state: State) -> Literal['change_after_user', '__end__']:
    if len(state['messages']) > 20:
        return '__end__'
    else:
        return 'change_after_user'


# --graph define


graph_builder = StateGraph(State)
graph_builder.add_node('llm', call_model)
graph_builder.add_node('user_response', user_response)
graph_builder.add_node('change_after_llm', change_status)
graph_builder.add_node('change_after_user', change_status)

graph_builder.add_edge(START, 'llm')
graph_builder.add_edge('llm', 'change_after_llm')
graph_builder.add_edge('change_after_llm', 'user_response')
graph_builder.add_conditional_edges('user_response', route,
    {
        '__end__': '__end__',
        'change_after_user': 'change_after_user'})
graph_builder.add_edge('change_after_user', 'llm')
graph = graph_builder.compile()

# call
# from PIL import Image
# from io import BytesIO
# Image.open(BytesIO(graph.get_graph().draw_mermaid_png())).save('graph.png')


initial_state = {
    'messages': [
        SystemMessage(content='你是一个智能助手，你认为{opinion}'),
        HumanMessage(content='你觉得 a等于几？')],
    'opinion': 'a = 1'}
graph.invoke(initial_state, config={'recursion_limit': 1000})
