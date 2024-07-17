import os
import json
from typing import TypedDict, Annotated
from rain_agent.configs import config

from langgraph.graph import StateGraph, START, END
from langgraph.graph import add_messages
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool

from .reader_prompts import *

# 定义状态
class State(TypedDict):
    messages: Annotated[list, add_messages]
    full_passage: str
    summaried_passage: str
    passage_exp: str
    passage_ref: str
    passage_intro: str


# llm_core负责核心逻辑，llm_long负责对长文本处理
llm_core = ChatOpenAI(
    model='qwen-max',
    openai_api_key=config['key']['bailian_api_key'],
    openai_api_base=config['key']['bailian_base_url']
)

llm_long = ChatOpenAI(
    model='qwen-long',
    openai_api_key=config['key']['bailian_api_key'],
    openai_api_base=config['key']['bailian_base_url'],
)


class RefToolNode:
    """A node that runs the tools requested in the last AIMessage."""

    def __call__(self, inputs: dict):
        if messages := inputs.get("messages", []):
            message = messages[-1]
        else:
            raise ValueError("No message found in input")
        outputs = []
        tool_result = ''
        for tool_call in message.tool_calls:
            if tool_call['name'] == 'ref_tool':
                tool_result = ref_tool(tool_call['args'])
                outputs.append(
                    ToolMessage(
                        content=json.dumps(tool_result, ensure_ascii=False),
                        name=tool_call["name"],
                        tool_call_id=tool_call["id"],
                    )
                )
        return {"passage_ref": tool_result}


class ExpToolNode:
    """A node that runs the tools requested in the last AIMessage."""

    def __call__(self, inputs: dict):
        if messages := inputs.get("messages", []):
            message = messages[-1]
        else:
            raise ValueError("No message found in input")
        outputs = []
        tool_result = ''
        for tool_call in message.tool_calls:
            if tool_call['name'] == 'exp_tool':
                tool_result = exp_tool(tool_call['args'])
                outputs.append(
                    ToolMessage(
                        content=json.dumps(tool_result, ensure_ascii=False),
                        name=tool_call["name"],
                        tool_call_id=tool_call["id"],
                    )
                )
        return {"passage_exp": tool_result}


class IntroToolNode:
    """A node that runs the tools requested in the last AIMessage."""

    def __call__(self, inputs: dict):
        if messages := inputs.get("messages", []):
            message = messages[-1]
        else:
            raise ValueError("No message found in input")
        outputs = []
        tool_result = ''
        for tool_call in message.tool_calls:
            if tool_call['name'] == 'intro_tool':
                tool_result = exp_tool(tool_call['args'])
                outputs.append(
                    ToolMessage(
                        content=json.dumps(tool_result, ensure_ascii=False),
                        name=tool_call["name"],
                        tool_call_id=tool_call["id"],
                    )
                )
        return {"passage_intro": tool_result}


@tool
def ref_tool(passage: str):
    """该工具能够找出论文中的关键引用文献"""
    ref_chain = ref_prompt_template | llm_long
    tool_result = ref_chain.invoke({'passage': passage})
    return tool_result.content


@tool
def exp_tool(passage: str):
    """该工具能对论文的实验章节进行提炼总结"""
    exp_chain = exp_prompt_template | llm_long
    tool_result = exp_chain.invoke({'passage': passage})
    return tool_result.content


@tool
def intro_tool(passage: str):
    """该工具能够对论文的介绍（introduction）章节进行提炼总结，找出其中的关键信息"""
    intro_chain = intro_prompt_template | llm_long
    tool_result = intro_chain.invoke({'passage': passage})
    return tool_result.content


# 定义不同的node
def summary_bot(state: State):
    result_state: state = {
        'messages': state['messages'],  # might never be used
        'full_passage': state['full_passage'],  # should already be there
        'summaried_passage': '',
        'passage_exp': '',
        'passage_ref': '',
        'passage_intro': ''
    }
    summary_chain = summary_prompt_template | llm_long
    summary_result = summary_chain.invoke(
        {
            'messages': [
                SystemMessage(content=state['full_passage']),
                HumanMessage(content='请总结这篇论文')]
        }
    ).content
    result_state['summaried_passage'] = summary_result

    llm_control = llm_core.bind_tools([exp_tool, ref_tool, intro_tool])
    summary_chain = control_prompt_template | llm_control

    res = summary_chain.invoke(
        {'messages': [HumanMessage(content=f'请帮我找出论文的重点引用文献:\n\n{summary_result}')]})
    res2 = summary_chain.invoke({'messages': [HumanMessage(content=f"请帮我总结论文的实验章节:\n\n{summary_result}")]})
    res3 = summary_chain.invoke({'messages': [HumanMessage(content=f"请帮我总结论文的Introduction章节:\n\n{summary_result}")]})

    res.tool_calls.extend(res2.tool_calls + res3.tool_calls)
    for e in res.tool_calls:
        if 'passage' in e['args']:
            e['args']['passage'] = state['full_passage']
    result_state['messages'] = [res]
    return result_state


def gather_bot(state: State):
    final = f"""
# Summary
    
{state['summaried_passage']}

# Introduction

{state['passage_intro']}

# Exp
    
{state['passage_exp']}

# Ref
    
{state['passage_ref']}
    """

    return {'summaried_passage': final}


graph_builder = StateGraph(State)

graph_builder.add_node('summary_bot', summary_bot)
graph_builder.add_node('ref_tool', RefToolNode())
graph_builder.add_node('exp_tool', ExpToolNode())
graph_builder.add_node('intro_tool', IntroToolNode())
graph_builder.add_node('gather_bot', gather_bot)

graph_builder.add_edge(START, 'summary_bot')
graph_builder.add_edge('summary_bot', 'ref_tool')
graph_builder.add_edge('summary_bot', 'exp_tool')
graph_builder.add_edge('summary_bot', 'intro_tool')
graph_builder.add_edge('ref_tool', 'gather_bot')
graph_builder.add_edge('exp_tool', 'gather_bot')
graph_builder.add_edge('intro_tool', 'gather_bot')
graph_builder.add_edge('gather_bot', END)
graph = graph_builder.compile()