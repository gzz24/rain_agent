"""
测试condition
如何用尽可能少的代码去构造一个可运行的带 condition 的图？
"""
from io import BytesIO
from PIL import Image
import asyncio
from typing import TypedDict, Annotated, Literal
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import ToolMessage
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import tools_condition, ToolNode


class State(TypedDict):
    messages: Annotated[list, add_messages]


@tool
def artist(theme: str) -> str:
    """paint for the theme"""
    print(f'Paint {theme}')
    return 'can you draw another painting about school for me?'


# def run_tool(state: State):
#     messages = state['messages'][-1]
#     tool_args = messages.tool_calls[0]['args']
#     tool_name = messages.tool_calls[0]['name']
#     tool_id = messages.tool_calls[0]['id']
#     tool_result = artist.invoke(tool_args)
#     return {'messages': [ToolMessage(content=tool_result, name=tool_name, tool_call_id=tool_id)]}


llm = ChatOpenAI(
    model='qwen-max',
    openai_api_key='sk-9c3717935de94e339b10435852fe760e',
    openai_api_base='https://dashscope.aliyuncs.com/compatible-mode/v1',
    streaming=True
).bind_tools([artist])

async def bot(state):
    s = state['messages']
    for e in s:
        if e.content == '':
            e.content = '-'
    res = await llm.ainvoke(s)
    ret = {'messages': [res]}
    return ret

def route(state):
    if isinstance(state, list):
        ai_msg = state[-1]
    elif msg := state.get('messages', []):
        ai_msg = msg[-1]
    else:
        raise AssertionError
    if hasattr(ai_msg, 'tool_calls') and len(ai_msg.tool_calls) > 0:
        return 'tools'
    else:
        return '__end__'

graph_builder = StateGraph(State)
graph_builder.add_node('bot', bot)
graph_builder.add_node('tools', ToolNode(tools=[artist]))

# graph_builder.add_conditional_edges('bot', tools_condition)
graph_builder.add_edge(START, 'bot')
graph_builder.add_edge('bot', END)
graph_builder.add_edge('tools', 'bot')
graph_builder.add_conditional_edges('bot', route)
graph = graph_builder.compile()


# Image.open(BytesIO(graph.get_graph().draw_mermaid_png())).save('condition.png')
# print(graph.invoke({'messages': 'hi, can you draw a painting about school for me?'}))


async def run_graph():
    inputs = {'messages': 'hi, can you draw a painting about school for me?'}
    async for event in graph.astream_events(inputs, version='v1'):
        kind = event['event']
        if kind == "on_chat_model_stream":
            content = event["data"]["chunk"].content
            if content:
                # Empty content in the context of OpenAI or Anthropic usually means
                # that the model is asking for a tool to be invoked.
                # So we only print non-empty content
                print(content, end="|", flush=True)
        elif kind == "on_tool_start":
            print("--", flush=True)
            print(
                f"Starting tool: {event['name']} with inputs: {event['data'].get('input')}",
                flush=True
            )
        elif kind == "on_tool_end":
            print(f"Done tool: {event['name']}", flush=True)
            print(f"Tool output was: {event['data'].get('output')}", flush=True)
            print("--", flush=True)


asyncio.run(run_graph())