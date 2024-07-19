import json
from typing import TypedDict, Annotated, List, Dict
from langchain_core.pydantic_v1 import BaseModel, Field
from langgraph.graph import START, END, StateGraph

from rain_agent.agent_apps.paper_reader.v2.reader_prompts import *
from rain_agent.agent_apps.paper_reader.v2.reader_llm import llm_core, llm_long


class SummaryState(TypedDict):
    paper_content: str

    summary: str


def summary(state: SummaryState):
    passage = state['paper_content']

    summary_llm = control_prompt_template | llm_long

    res = summary_llm.invoke({'passage': passage})

    return {'summary': res.content}


graph_builder = StateGraph(SummaryState)
graph_builder.add_node('summary_node', summary)
graph_builder.add_edge(START, 'summary_node')
graph_builder.add_edge('summary_node', END)
summary_graph = graph_builder.compile()