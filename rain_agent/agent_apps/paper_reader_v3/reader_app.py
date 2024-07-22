import base64
import os
import glob
import requests
from io import BytesIO
from typing import List, Dict
from PIL import Image

import streamlit as st
from rain_agent.configs import config


working_dir = '/Users/zhoushuzhe1/code/rain_agent/data/working_dir/'
working_user = 'giere'
working_space = 'base'
working_path = os.path.join(working_dir, working_user, working_space)
if not os.path.exists(working_path):
    os.makedirs(working_path)

# dir -> user -> space
# under space:
# - paper [by_name]
# - others*


@st.cache_data
def _get_pdf_blocks(pdf_path) -> List[List[dict]]:
    # reader = PdfReader(pdf_path)
    #
    # text_lst = []
    # for e in reader.pages:
    #     text_lst.append(e.extract_text())
    #
    # text = '\n\n'.join(text_lst)

    with open(pdf_path, 'rb') as f:
        files = {'file': ('_temp.pdf', f, 'application/pdf')}
        response = requests.post(config['pdf']['url'], files=files, timeout=60)

    try:
        res_json = response.json()
    except:
        print(response.text)
        raise Exception
    # text = ''
    # for e_page in res_json:
    #     for e_block in e_page:
    #         text += e_block['text']
    #     text += '\n\n'
    #
    # return text
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


def run_reader():
    fname = _upload()

    if st.button(label='开始读论文'):

        pdf_json = _get_pdf_blocks(fname)

        for i, e_page in enumerate(pdf_json):
            st.write(f'## Page {i+1}')
            for e_block in e_page:
                if e_block['layout_type'] not in {'Figure', 'Table', 'Equation'}:
                    st.write(e_block['text'])
                else:
                    img_data = base64.b64decode(e_block['mm'])
                    buffer = BytesIO(img_data)
                    img = Image.open(buffer)
                    st.image(img)
            st.divider()




