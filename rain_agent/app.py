import os
import glob
import uuid
from configs import config
#os.environ['LANGCHAIN_TRACING_V2'] = 'true'
#os.environ['LANGCHAIN_API_KEY'] = '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'
import streamlit as st

from agent_apps.naive_chat import run_naive_chat

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
    run_naive_chat(config)
else:
    st.write('开发中...')