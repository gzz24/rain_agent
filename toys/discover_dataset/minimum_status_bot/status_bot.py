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

class State2(TypedDict):
    messages: Annotated[list, add_messages]


# -- llm and tool
llm = ChatOpenAI(
    model='qwen-max',
    openai_api_key=config['key']['bailian_api_key'],
    openai_api_base=config['key']['bailian_base_url']
)


def call_model(state: State):
    messages = state['messages']
    response = llm.invoke(messages)
    print(f"model: {response.content}")
    return {
        'messages': [response]
    }


def status_update(state: State):
    pass


def status_apply(state: State):
    pass


def route(state: State):
    pass

# --graph define


graph_builder = StateGraph(State)
graph_builder.add_node('llm', call_model)
graph_builder.add_node('status_apply', status_apply)
graph_builder.add_node('status_update', status_update)

graph_builder.add_edge(START, 'status_apply')
graph_builder.add_edge('status_apply', 'llm')
graph_builder.add_edge('llm', 'status_update')
graph_builder.add_edge('status_update', END)
graph = graph_builder.compile()


def foo(state: State):
    pass

def bar(state: State):
    pass

graph_builder2 = StateGraph(State2)
graph_builder2.add_node('node1', foo)
graph_builder2.add_node('node2', foo)
graph_builder2.add_node('node3', foo)
graph_builder2.add_node('graph', graph)

graph_builder2.add_edge(START, 'node1')
graph_builder2.add_edge('node1', 'graph')
graph_builder2.add_edge('graph', 'node2')
graph_builder2.add_edge('node2', 'node3')
graph_builder2.add_edge('node3', END)

graph2 = graph_builder2.compile()





class sql_State(TypedDict):
    messages: Annotated[list, add_messages]


llm = ChatOpenAI(
    model='qwen-max',
    openai_api_key=config['key']['bailian_api_key'],
    openai_api_base=config['key']['bailian_base_url']
)

def call_model(state: sql_State):
    messages = state['messages']
    response = llm.invoke(messages)
    print(f"model: {response.content}")
    return {
        'messages': [response]
    }


def status_update(state: sql_State):
    pass


def status_apply(state: sql_State):
    pass

graph_builder = StateGraph(sql_State)
graph_builder.add_node('llm', call_model)
graph_builder.add_node('status_apply', status_apply)
graph_builder.add_node('status_update', status_update)

graph_builder.add_edge(START, 'status_apply')
graph_builder.add_edge('status_apply', 'llm')
graph_builder.add_edge('llm', 'status_update')
graph_builder.add_edge('status_update', END)
graph = graph_builder.compile()


def call_sql_status_graph(outer_state):
    sub_state_name = 'sql_'
    sub_states = outer_state['sub_states']
    sub_state = sub_states[sub_state_name]

    # don't forget to use response
    new_sub_state, response = graph.invoke(sub_state)

    sub_states[sub_state_name] = new_sub_state

    return {
        'sub_states': sub_states
    }



# call
from PIL import Image
from io import BytesIO
Image.open(BytesIO(graph.get_graph().draw_mermaid_png())).save('graph.png')
#

# initial_state = {
#     'messages': [
#         SystemMessage(),
#         HumanMessage()]
# }
# graph.invoke(initial_state, config={'recursion_limit': 1000})
