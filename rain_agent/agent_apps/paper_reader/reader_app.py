import os
import glob
import json
from typing import TypedDict, Annotated
from rain_agent.configs import config
from PyPDF2 import PdfReader
from streamlit_pdf_viewer import pdf_viewer

import streamlit as st

from .reader_agent import *

pdf_dir = '/Users/zhoushuzhe1/code/rain_agent/data/papers'


def _get_pdf_text(pdf_path) -> str:
    # text_path = pdf_path[:-4] + '.txt'
    # with open(text_path, 'r', encoding='utf-8') as f:
    #     t = f.read().strip()

    reader = PdfReader(pdf_path)

    text_lst = []
    for e in reader.pages:
        text_lst.append(e.extract_text())

    text = '\n\n'.join(text_lst)
    return text


def _st_select_pdf_file():
    pdf_files_full = list(glob.glob(os.path.join(pdf_dir, '*.pdf')))
    pdf_file2name = {x: os.path.basename(x) for x in pdf_files_full}
    pdf_name2file = {v: k for k, v in pdf_file2name.items()}
    selected_pdf = st.selectbox('论文选择', pdf_name2file.keys())
    return selected_pdf, pdf_file2name, pdf_name2file


def run_reader():
    # select paper
    selected_pdf, f2n, n2f = _st_select_pdf_file()

    pdf_texts = _get_pdf_text(n2f[selected_pdf])

    if st.button(label='开始读论文'):

        initial_state: State = {
            'messages': [],
            'summaried_passage': '',
            'full_passage': pdf_texts,
            'passage_exp': '',
            'passage_ref': '',
            'passage_intro': ''
        }

        res = graph.invoke(initial_state)

        print(json.dumps(res['summaried_passage'], ensure_ascii=False))
        st.write(res['summaried_passage'])
        st.divider()
        st.code(res['summaried_passage'])