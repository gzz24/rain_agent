import os
import glob
import uuid
import sys

import streamlit as st

sys.path.append('..')
from typing import Annotated
from typing_extensions import TypedDict

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_openai import ChatOpenAI


class State(TypedDict):
    messages: Annotated[list, add_messages]


def run_naive_agent(config):
    llm = ChatOpenAI(
        model='qwen-max',
        openai_api_key=config['key']['bailian_api_key'],
        openai_api_base=config['key']['bailian_base_url']
    )
    graph_builder = StateGraph(State)

    def chatbot(state: State):
        return {
            'messages': [llm.invoke(state['messages'])]
        }

    graph_builder.add_node('chatbot', chatbot)
    graph_builder.add_edge(START, 'chatbot')
    graph_builder.add_edge('chatbot', END)
    graph = graph_builder.compile()

    if prompt := st.chat_input('input your prompt'):
        with st.chat_message('user'):
            st.markdown(prompt)

        response = graph.invoke({'messages': ('user', prompt)})

        with st.chat_message('assistant'):
            st.markdown(response['messages'][-1].content)

