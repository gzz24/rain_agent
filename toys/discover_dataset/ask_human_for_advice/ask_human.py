# 固化代码
import json
from typing import List, Dict, Literal, Union

from configs import config
from tools import *
from langchain.globals import set_verbose, set_debug

from langgraph.graph import START, END, StateGraph, add_messages
from typing import TypedDict, List, Dict, Annotated
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import ToolNode
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage


# end


# 固化模板

# -- graph
class State(TypedDict):
    messages: Annotated[list, add_messages]


# -- llm and tool
llm = ChatOpenAI(
    model='qwen-max',
    openai_api_key=config['key']['bailian_api_key'],
    openai_api_base=config['key']['bailian_base_url']
)
tools = [
    check_memory,
    check_file_content,
    get_full_file_content,
    check_past_error,
    check_if_file_readable,
    check_weather,
    get_file_size
]
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
        print(f"call tool: {e['name']}")
        res = globals()[e['name']]()
        messages.append(
            ToolMessage(
                content=str(res),
                name=e['name'],
                id='call' + e['id'],
                tool_call_id=e['id']
            )
        )

    return {
        'messages': messages[:1]
    }


def user_decision(state: State):
    messages = state['messages']
    if messages[-1].content:
        print(messages[-1].content)
    else:
        print(messages[-1].tool_calls)

    human_query = input('我的建议:')
    return {
        'messages': [
            HumanMessage(content=human_query)
        ]
    }


# -- control
def route(state: State) -> Literal['tool_node', 'llm', '__end__', 'user_decision']:
    messages = state['messages']
    last_message = messages[-1]
    if last_message.tool_calls:
        return 'tool_node'
    return 'user_decision'

def route2(state: State):
    messages = state['messages']
    last_message = messages[-1]

    if last_message.content == 'exit':
        return '__end__'
    else:
        return 'llm'


# --graph define


graph_builder = StateGraph(State)
graph_builder.add_node('llm', call_model)
graph_builder.add_node('tool_node', call_tool)
graph_builder.add_node('user_decision', user_decision)
graph_builder.add_edge(START, 'llm')
graph_builder.add_conditional_edges('llm', route, {'user_decision': 'user_decision', 'tool_node': 'tool_node'})
graph_builder.add_conditional_edges('user_decision', route2, {'__end__': '__end__', 'llm': 'llm'})
graph_builder.add_edge('tool_node', 'llm')
graph = graph_builder.compile()

# from PIL import Image
# from io import BytesIO
# Image.open(BytesIO(graph.get_graph().draw_mermaid_png())).save('graph.png')

# call

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
