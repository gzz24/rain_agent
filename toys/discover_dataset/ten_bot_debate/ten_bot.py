# 固化代码
import json
from typing import List, Dict, Literal, Union

from configs import config, people_lst, stands
from prompts import *
from tools import *
from langchain.globals import set_verbose, set_debug

from langgraph.graph import START, END, StateGraph, add_messages
from typing import TypedDict, List, Dict, Annotated
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import ToolNode
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage, trim_messages, AIMessage
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
    # 所有发言记录
    debate: List[Dict[str, str]]
    # 每个人独有的 message history
    personal_messages: Dict[str, List]

# -- llm and tool
llm = ChatOpenAI(
    model='qwen-max',
    openai_api_key=config['key']['bailian_api_key'],
    openai_api_base=config['key']['bailian_base_url'],
    temperature=0.9
)


def call_model(state: State):
    name = people_lst[(len(state['debate']) - 1) % 10]
    stand = stands[(len(state['debate']) - 1) % 5]
    debate = state['debate']
    history_msgs = state['personal_messages'][name]

    cur_llm = prompt_template | llm

    # build history
    new_debate_record_lst = []
    for e in debate[::-1]:
        if e['name'] == name:
            break
        new_debate_record_lst.append(e)
    debate_record_content = ''
    for e in new_debate_record_lst:
        debate_record_content += f"{e['name']}: {e['speech']}"
    history_msgs.append(HumanMessage(
        content=human_prompt.format(
            history_speech=debate_record_content,
            person=name,
            stand=stand)))

    # model
    response = ''
    print(f"{name}: ", end='')
    for chunk in cur_llm.stream({'person': name, 'messages': history_msgs}):
        res = chunk.content
        response += res
        print(res, end='', flush=True)
    print('\n' + '-'*20)

    # store output
    debate.append({
        'name': name,
        'speech': response
    })

    # update state
    all_personal_msgs = state['personal_messages']
    all_personal_msgs[name] = history_msgs + [AIMessage(content=response)]
    new_state: State = {
        'debate': debate,
        'personal_messages': all_personal_msgs
    }
    return new_state


# -- control
def route(state: State):
    debate = state['debate']
    if len(debate) > 300:
        return '__end__'
    else:
        return people_lst[0]


# --graph define


graph_builder = StateGraph(State)

for e_name in people_lst:
    graph_builder.add_node(e_name, lambda x: call_model(x))

graph_builder.add_edge(START, people_lst[0])
for i in range(len(people_lst) - 1):
    graph_builder.add_edge(people_lst[i], people_lst[i + 1])
graph_builder.add_conditional_edges(people_lst[-1], route, {'__end__': '__end__', people_lst[0]: people_lst[0]})
graph = graph_builder.compile()

# # call
# from PIL import Image
# from io import BytesIO
# Image.open(BytesIO(graph.get_graph().draw_mermaid_png())).save('graph.png')


initial_state: State = {
    'debate': [
        {'name': '裁判', 'speech': '辩论的内容是：电商是否会取代线下购物？请开始你们的辩论'}
    ]
}
personal_msgs = {n: [] for n in people_lst}
initial_state['personal_messages'] = personal_msgs

graph.invoke(initial_state, config={'recursion_limit': 1000})


"""
无聊的车轱辘话

LLM 目前还无法产生新的独特的见解，只能作为人的思考的延伸
"""