import json
from typing import List, Dict, Literal

from configs import config
from tools import *
from langchain.globals import set_verbose, set_debug

from langgraph.graph import START, END, StateGraph, add_messages
from typing import TypedDict, List, Dict, Annotated
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import ToolNode
from langchain_core.messages import HumanMessage, SystemMessage


class State(TypedDict):
    messages: Annotated[List[str], add_messages]
    info: List[str]


llm = ChatOpenAI(
    model='qwen-max',
    openai_api_key=config['key']['bailian_api_key'],
    openai_api_base=config['key']['bailian_base_url']
)

tools = [
    get_dataset_files, get_folder_files,
    get_file_basic_info,
    get_json_info, get_json_part_content,
    get_sql_info, get_sql_part_content]

llm_with_tool = llm.bind_tools(tools)

tool_node = ToolNode(tools)


def route(state: State) -> Literal['tool_node', 'llm', '__end__']:
    messages = state['messages']
    last_message = messages[-1]
    if last_message.tool_calls:
        return 'tool_node'
    return '__end__'


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


graph_builder = StateGraph(State)
graph_builder.add_node('llm', call_model)
graph_builder.add_node('tool_node', tool_node)
graph_builder.add_edge(START, 'llm')
graph_builder.add_conditional_edges('llm', route)
graph_builder.add_edge('tool_node', 'llm')
graph = graph_builder.compile()


initial_state = {
    'messages': [
        SystemMessage(content="""
你是一个资深机器学习工程师，你将负责对数据集的规模、分布、类型、文件等信息进行收集和分析。

- 你将持续结合工具输出进行分析
- 请不要让用户为你做决策，比如让用户选择需要分析的数据集文件等。你可以直接自主规划
- 请始终使用中文
        """),
        HumanMessage(content='请你帮我探索一下Spider数据集')]
}
graph.invoke(initial_state)
