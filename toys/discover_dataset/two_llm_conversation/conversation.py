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
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage, AIMessage
from langchain_core.messages import trim_messages

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

# end


# 固化模板

# -- graph
class State(TypedDict):
    A_messages: Annotated[list, add_messages]
    B_messages: Annotated[list, add_messages]


# -- llm and tool
llm = trimmer | ChatOpenAI(
    model='qwen-max',
    openai_api_key=config['key']['bailian_api_key'],
    openai_api_base=config['key']['bailian_base_url']
)


def call_model_A(state: State):
    messages = state['A_messages']
    print("model A:", end='')
    response = ''
    for chunk in llm.stream(messages):
        res = chunk.content
        print(res, end='', flush=True)
        response += res

    print('\n')

    return {
        'A_messages': [
            AIMessage(content=response)
        ]
    }


def invert_A(state: State):
    # the last message must be assistant
    A_messages = state['A_messages']

    result = [
        HumanMessage(content=A_messages[-1].content)
    ]
    return {
        'B_messages': result
    }


def call_model_B(state: State):
    messages = state['B_messages']
    print("model B:", end='')
    response = ''
    for chunk in llm.stream(messages):
        res = chunk.content
        print(res, end='', flush=True)
        response += res
    print('\n')

    return {
        'B_messages': [
            AIMessage(content=response)
        ]
    }


def invert_B(state: State):
    # the last message must be assistant
    B_messages = state['B_messages']

    result = [
        HumanMessage(content=B_messages[-1].content)
    ]
    return {
        'A_messages': result
    }


def route(state: State):
    if len(state['A_messages']) > 50:
        return '__end__'
    else:
        return 'chat_A'


# --graph define

graph_builder = StateGraph(State)
graph_builder.add_node('chat_A', call_model_A)
graph_builder.add_node('chat_B', call_model_B)
graph_builder.add_node('invert_A', invert_A)
graph_builder.add_node('invert_B', invert_B)

graph_builder.add_edge(START, 'chat_A')
graph_builder.add_edge('chat_A', 'invert_A')
graph_builder.add_edge('invert_A', 'chat_B')
graph_builder.add_edge('chat_B', 'invert_B')
graph_builder.add_conditional_edges('invert_B', route, {'__end__': '__end__', 'chat_A': 'chat_A'})
graph = graph_builder.compile()

# call
# from PIL import Image
# from io import BytesIO
# Image.open(BytesIO(graph.get_graph().draw_mermaid_png())).save('graph.png')
#

initial_state = {
    'A_messages': [
        HumanMessage(content='请你帮我探索一下给你的 json obj 。你可以通过工具对该 obj 进行观察')]
}
graph.invoke(initial_state, config={'recursion_limit': 1000})
