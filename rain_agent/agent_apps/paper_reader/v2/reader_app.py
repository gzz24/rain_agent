import os
import glob
import streamlit as st
import requests
import json

from PyPDF2 import PdfReader
from .reader_main_agent import main_graph
from .reader_ref_agent import ref_graph
from .reader_summary_agent import summary_graph

from rain_agent.configs import config

pdf_dir = '/Users/zhoushuzhe1/code/rain_agent/data/papers'


@st.cache_data
def _get_pdf_text(pdf_path) -> str:
    # reader = PdfReader(pdf_path)
    #
    # text_lst = []
    # for e in reader.pages:
    #     text_lst.append(e.extract_text())
    #
    # text = '\n\n'.join(text_lst)

    with open(pdf_path, 'rb') as f:
        files = {'file': ('_temp.pdf', f, 'application/pdf')}
        response = requests.post(config['pdf']['url'], files=files)

    res_json = response.json()
    text = ''
    for e_page in res_json:
        for e_block in e_page:
            text += e_block['text']
        text += '\n\n'

    return text


def _st_select_pdf_file():
    pdf_files_full = list(glob.glob(os.path.join(pdf_dir, '*.pdf')))
    pdf_file2name = {x: os.path.basename(x) for x in pdf_files_full}
    pdf_name2file = {v: k for k, v in pdf_file2name.items()}
    selected_pdf = st.selectbox('论文选择', pdf_name2file.keys())
    return selected_pdf, pdf_file2name, pdf_name2file



def run_reader():
    selected_pdf, f2n, n2f = _st_select_pdf_file()

    if st.button(label='开始读论文'):

        pdf_texts = _get_pdf_text(n2f[selected_pdf])
        # summary
        summary_state = {'paper_content': pdf_texts}
        print('summary...')
        res = summary_graph.invoke(summary_state)
        with st.expander('Summary'):
            st.write(res['summary'])

        # main
        main_state = {'paper_content': pdf_texts}
        print('main...')
        res2 = main_graph.invoke(main_state)
        with st.expander('Main'):
            points, details = res2['point'], res2['point_detail']
            for ep, ed in zip(points, details):
                st.subheader(ep)
                st.write(ed)
                st.divider()

        # ref
        ref_state = {'paper_content': pdf_texts}
        print('ref...')
        res3 = ref_graph.invoke(ref_state)
        with st.expander('Ref'):
            ref, ref_content = res3['ref'], res3['ref_content']
            for er, rc in zip(ref, ref_content):
                st.write(er.year)
                st.write(er.authors)
                st.write(er.title)
                st.write(er.how_related)
                for erc in rc:
                    st.text(erc.location)
                    st.code(erc.origin_text)
                    st.text(erc.ref_reason)
                    st.text('---\n---')
                st.divider()