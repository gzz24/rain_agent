import os
import glob
from configs import config

import streamlit as st
from openai import OpenAI
from langchain_openai import ChatOpenAI


# 暂时只有这些应用
op_lst = ['naive_chat', 'other_agent']
st.set_page_config(
    page_title='rain_agent',
    initial_sidebar_state='expanded'
)


# page config L1
with st.sidebar:
    selected_agent = st.selectbox('我的Agent', op_lst, placeholder=op_lst[0])

if selected_agent == 'naive_chat':
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    with st.sidebar:
        clear_his = st.button('清除对话历史')
        if clear_his:
            st.session_state.messages = []
            st.rerun()  # tricky!

    llm = ChatOpenAI(
        model='qwen-max',
        openai_api_key=config['key']['bailian_api_key'],
        openai_api_base=config['key']['bailian_base_url']
    )

    if prompt := st.chat_input('input your prompt'):
        with st.chat_message('user'):
            st.markdown(prompt)
        st.session_state.messages.append({
            'role': 'user',
            'content': prompt
        })

        response = llm.stream(prompt)
        def stream_response(res, store):
            for chunk in res:
                store.append(chunk.content)
                yield chunk.content

        with st.chat_message('assistant'):
            temp_store = []
            st.write_stream(stream_response(response, temp_store))
            response_str = ''.join(temp_store)
        st.session_state.messages.append({
            'role': 'assistant',
            'content': response_str
        })

else:
    st.write('开发中...')