import json
from typing import TypedDict, Annotated, List, Dict
from langchain_core.pydantic_v1 import BaseModel, Field
from langgraph.graph import START, END, StateGraph

from rain_agent.agent_apps.paper_reader.v2.reader_prompts import *
from rain_agent.agent_apps.paper_reader.v2.reader_llm import llm_core, llm_long


class MainState(TypedDict):
    paper_content: str

    point: List[str]
    point_detail: List[str]


class Point(BaseModel):
    """该论文的主要工作"""
    points: List[str] = Field(description='论文的主要贡献/主要工作的点的列表')


def main_intro(state: MainState):
    passage = state['paper_content']

    main_llm = intro_prompt_template | llm_long
    res = main_llm.invoke({'passage': passage})

    extract_llm = intro_extract_prompt_template | llm_core.with_structured_output(Point)

    extracted = extract_llm.invoke({'summary': res.content})

    return {
        'point': extracted.points
    }


def main_detail(state: MainState):
    passage = state['paper_content']

    detail_llm = main_prompt_template | llm_long

    input_lst = []
    for e in state['point']:
        input_lst.append({
            'passage': passage,
            'point': e
        })
    details = detail_llm.batch(input_lst, config={'max_concurrency': 5})

    return {
        'point_detail': list(x.content for x in details)
    }


graph_builder = StateGraph(MainState)
graph_builder.add_node('intro', main_intro)
graph_builder.add_node('detail', main_detail)
graph_builder.add_edge(START, 'intro')
graph_builder.add_edge('intro', 'detail')
graph_builder.add_edge('detail', END)
main_graph = graph_builder.compile()
