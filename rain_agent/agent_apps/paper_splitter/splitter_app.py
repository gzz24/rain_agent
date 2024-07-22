import os
import requests
from typing import List, Dict

import streamlit as st

from rain_agent.agent_apps.paper_splitter.splitter_agent import *
from rain_agent.configs import config

working_dir = '/Users/zhoushuzhe/Code/rain_agent/data/splitter/working_dir/'
working_user = 'giere'
working_space = 'base'
working_path = os.path.join(working_dir, working_user, working_space)
if not os.path.exists(working_path):
    os.makedirs(working_path)


@st.cache_data
def _get_pdf_blocks(pdf_path) -> List[List[dict]]:
    with open(pdf_path, 'rb') as f:
        files = {'file': ('_temp.pdf', f, 'application/pdf')}
        response = requests.post(config['pdf']['url'], files=files, timeout=60)

    try:
        res_json = response.json()
    except:
        print(response.text)
        raise Exception
    return res_json


def _upload():
    uploaded_file = st.file_uploader('请上传一篇pdf格式的论文', type='pdf')
    if uploaded_file is not None:
        bytes_data = uploaded_file.getvalue()
        paper_name = uploaded_file.name.split('.')[0]
        if not os.path.exists(os.path.join(working_path, 'paper', paper_name)):
            os.makedirs(os.path.join(working_path, 'paper', paper_name))
        with open(os.path.join(working_path, 'paper', paper_name, 'paper.pdf'), 'wb') as f:
            f.write(bytes_data)

        return os.path.join(working_path, 'paper', paper_name, 'paper.pdf')
    else:
        raise Exception


def run_splitter():
    fname = _upload()

    if st.button(label='开始切分'):
        pdf_json = _get_pdf_blocks(fname)
        text_blocks = []
        for e_page in pdf_json:
            for e_block in e_page:
                if len(e_block['text']) != 0:
                    text_blocks.append(e_block['text'])
                else:
                    text_blocks.append(f"[{e_block['layout_type']}]")

        state = {
            'passage_blocks': text_blocks,
            'current_idx': 0,
            'titles': []
        }

        res = graph.invoke(state, config={'recursion_limit': 1000})

        last_title = ''
        for e_title, e_content in zip(res, text_blocks):
            if e_title == last_title:
                st.write(e_content)
            else:
                st.divider()
                st.header(e_title)
                st.write(e_content)
            last_title = e_title