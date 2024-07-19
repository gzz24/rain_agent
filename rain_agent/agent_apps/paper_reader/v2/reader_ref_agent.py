import json
from typing import TypedDict, Annotated, List, Dict
from langchain_core.pydantic_v1 import BaseModel, Field
from langgraph.graph import START, END, StateGraph

from rain_agent.agent_apps.paper_reader.v2.reader_prompts import *
from rain_agent.agent_apps.paper_reader.v2.reader_llm import llm_core, llm_long


class RefState(TypedDict):
    """
    引用分析Agent的state
    """
    paper_content: str

    raw_ref: str
    # 文章的所有关键引用文献
    ref: List[Dict[str, str]]
    # 文章引用文献关联的句子

    raw_ref_content: List[str]
    ref_content: List[List[Dict[str, str]]]


class Reference(BaseModel):
    """论文中一个引用文献的相关信息"""
    year: str = Field(description='该引用文献的发表年份')
    authors: str = Field(description='该引用文献的作者')
    title: str = Field(description='该引用文献的标题')
    how_related: str = Field(description='该引用文献与本文的详细关联。（请不要摘要该字段，请完整详细地抽取）')


class References(BaseModel):
    """论文中的关键引用文献的列表"""
    ref_lst: List[Reference] = Field(description='论文中的所有关键引用文献')


class RefReason(BaseModel):
    """论文中引用一篇文献的位置、片段、以及引用的具体原因"""
    location: str = Field(description='该论文引用这篇文献的章节位置')
    origin_text: str = Field(description='改论文引用这篇文献的原文片段')
    ref_reason: str = Field(description='该论文是如何引用这篇文献的')


class RefReasons(BaseModel):
    """论文中引用一篇文献的所有位置，对应的片段以及原因"""
    ref_reasons: List[RefReason] = Field(description='论文对一篇引用文献引用的所有位置、片段和原因')


def find_ref(state: RefState):
    """
    先用长文本模型总结论文的引用文献
    然后用普通模型将输出结构化
    :param state:
    :return:
    """
    print('finding')
    # 找到ref
    llm_finds_ref = ref_prompt_template | llm_long
    ref_found = llm_finds_ref.invoke({'passage': state['paper_content']}).content

    return {
        'raw_ref': ref_found
    }


def extract_ref(state: RefState):
    print('extract finds')

    ref_found = state['raw_ref']
    # 将ref结构化
    llm_extracts_ref = extract_ref_prompt_template | llm_core.with_structured_output(References)
    ref = llm_extracts_ref.invoke({'ref': ref_found})
    return {
        'ref': ref.ref_lst
    }


def relocate_ref(state: RefState):
    print('relocate')
    ref_lst = state['ref']
    relocate_llm = ref_relocate_prompt_template | llm_long

    input_lst = []
    for e_ref in ref_lst:
        input_lst.append({
            'passage': state['paper_content'],
            'year': e_ref.year, 'author': e_ref.authors, 'title': e_ref.title,
            'how_related': e_ref.how_related
        })

    ref_content = relocate_llm.batch(input_lst, config={'max_concurrency': 10})

    return {
        'raw_ref_content': list(x.content for x in ref_content)
    }


def extract_relocate_ref(state: RefState):
    print('extract relocate')
    raw_ref_content = state['raw_ref_content']

    extract_llm = extract_relocate_prompt_template | llm_core.with_structured_output(RefReasons)

    input_lst = [{'reason': x} for x in raw_ref_content]
    result = extract_llm.batch(input_lst, config={'max_concurrency': 10})

    return {
        'ref_content': list(x.ref_reasons for x in result)
    }


graph_builder = StateGraph(RefState)
graph_builder.add_node('find', find_ref)
graph_builder.add_node('extract_find', extract_ref)
graph_builder.add_node('relocate', relocate_ref)
graph_builder.add_node('extract_relocate', extract_relocate_ref)
graph_builder.add_edge(START, 'find')
graph_builder.add_edge('find', 'extract_find')
graph_builder.add_edge('extract_find', 'relocate')
graph_builder.add_edge('relocate', 'extract_relocate')
graph_builder.add_edge('extract_relocate', END)
ref_graph = graph_builder.compile()


# from io import BytesIO
# from PIL import Image
# Image.open(BytesIO(graph.get_graph().draw_mermaid_png())).save('ref.png')

# with open('/Users/zhoushuzhe1/code/rain_agent/data/papers/Spider- A Large-Scale Human-Labeled Dataset for Complex and Cross-Domain Semantic Parsing and Text-to-SQL Task.txt', 'r', encoding='utf-8') as f:
#     cont = f.read().strip()
#
# dummy_state = {'paper_content': cont}
#
# res = graph.invoke(dummy_state)
#
# import pickle
# pickle.dump(res, open('ref.pickle', 'wb'))
