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


# -- llm and tool
llm = trimmer | ChatOpenAI(
    model='qwen-max',
    openai_api_key=config['key']['bailian_api_key'],
    openai_api_base=config['key']['bailian_base_url']
)
tools = []
llm_with_tool = llm.bind_tools(tools)
tool_node = ToolNode(tools)


def call_model(state: State):
    messages = state['messages']
    for i in range(len(messages)):
        if messages[i].content == '':
            messages[i].content = '-'
    response = llm_with_tool.invoke(messages)
    print(f"model: {response.content}")
    return {
        'messages': [response]
    }


def call_tool(state: State):
    tool_call = state['messages'][-1].tool_calls

    messages = []
    for e in tool_call:
        res = globals()[e['name']](**e['args'])
        messages.append(
            ToolMessage(
                content=res,
                name=e['name'],
                id='call' + e['id'],
                tool_call_id=e['id']
            )
        )

    return {
        'messages': messages[:1]
    }


# -- control
def route(state: State) -> Literal['tool_node', 'llm', '__end__']:
    messages = state['messages']
    last_message = messages[-1]
    if last_message.tool_calls:
        return 'tool_node'
    return '__end__'



def status_update(state: State):
    pass


def status_apply(state: State):
    pass



# --graph define


graph_builder = StateGraph(State)
graph_builder.add_node('llm', call_model)
graph_builder.add_node('tool_node', call_tool)
graph_builder.add_edge(START, 'llm')
graph_builder.add_conditional_edges('llm', route)
graph_builder.add_edge('tool_node', 'llm')
graph = graph_builder.compile()

# call
# from PIL import Image
# from io import BytesIO
# Image.open(BytesIO(graph.get_graph().draw_mermaid_png())).save('graph.png')
#

initial_state = {
    'messages': [
        SystemMessage(content="""
你是一个资深 python 程序员，你有一系列接口

- 你将持续结合工具输出进行分析
- 请不要让用户为你做决策，比如让用户选择需要分析的数据集文件等。你可以直接自主规划
- 请始终使用中文
- json obj 已经存储在本地，你可以直接通过工具对其进行探索
- json 对象在此处可以为一个列表类型，你可以认为这个列表已经是一个 dict 的 value，然后你只需对该 value 进行探查
        """),
        HumanMessage(content='请你帮我探索一下给你的 json obj 。你可以通过工具对该 obj 进行观察')]
}
graph.invoke(initial_state, config={'recursion_limit': 1000})

