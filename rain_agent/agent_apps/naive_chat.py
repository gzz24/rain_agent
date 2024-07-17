import os
import glob
import uuid


import sys
sys.path.append('..')
import streamlit as st


from openai import OpenAI
from langchain_openai import ChatOpenAI
from langchain_core.chat_history import BaseChatMessageHistory, InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory


def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in st.session_state.store:
        st.session_state.store[session_id] = InMemoryChatMessageHistory()
    return st.session_state.store[session_id]


def stream_response(res, store):
    for chunk in res:
        store.append(chunk.content)
        yield chunk.content

def run_naive_chat(config):
    # history for streamlit
    if 'messages' not in st.session_state:
        st.session_state.messages = []
        conversation_uuid = str(uuid.uuid4())
        st.session_state.uuid = conversation_uuid
        st.session_state.store = {}
    # re-render history conversation
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # clear button
    with st.sidebar:
        clear_his = st.button('清除对话历史')
        if clear_his:
            st.session_state.messages = []
            st.session_state.uuid = str(uuid.uuid4())
            st.session_state.store = {}
            st.rerun()  # tricky!

    # chat instance
    llm = ChatOpenAI(
        model='qwen-max',
        openai_api_key=config['key']['bailian_api_key'],
        openai_api_base=config['key']['bailian_base_url']
    )
    with_message_history = RunnableWithMessageHistory(llm, get_session_history)
    config = {'configurable': {'session_id': st.session_state.uuid}}

    # get input
    if prompt := st.chat_input('input your prompt'):
        with st.chat_message('user'):
            st.markdown(prompt)
        st.session_state.messages.append({
            'role': 'user',
            'content': prompt
        })

        response = with_message_history.stream(
            prompt,
            config=config
        )
        with st.chat_message('assistant'):
            temp_store = []
            st.write_stream(stream_response(response, temp_store))
            response_str = ''.join(temp_store)
        st.session_state.messages.append({
            'role': 'assistant',
            'content': response_str
        })
