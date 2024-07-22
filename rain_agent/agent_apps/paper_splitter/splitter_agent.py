import json
from typing import TypedDict, Annotated, List, Dict
from operator import add
from openai import BadRequestError

from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langgraph.graph import add_messages, START, END, StateGraph

from rain_agent.configs import config
config = {
    'key': {
        'bailian_api_key': 'sk-9c3717935de94e339b10435852fe760e',
        'bailian_base_url': 'https://dashscope.aliyuncs.com/compatible-mode/v1'
    }
}

# Status

class SplitState(TypedDict):
    passage_blocks: List[str]  # static
    current_idx: Annotated[int, add]
    titles: List[str]


class Title(BaseModel):
    """文本片段的标题。一级标题以一个“#”开头，二级标题则是“##”，依此类推"""
    title_str: str = Field(description="文本片段的标题，markdown格式")


# Models
llm_core = ChatOpenAI(
    model='qwen-max',
    openai_api_key=config['key']['bailian_api_key'],
    openai_api_base=config['key']['bailian_base_url']
)


# Prompts
tt_sys_prompt = """
你是一个论文阅读助手。你会接收到一些来自论文中的文本，这些文本因为某些原因丢失了标题的格式，而你需要结合上下文，重新为每个段落划分其对应的标题章节
"""
tt_text_prompt = """
{title3}
{pre3}

{title2}
{pre2}

{title1}
{pre1}

{current}

{post1}

{post2}

{post3}
"""
tt_user_prompt = """
"""
tt_prompt_template = ChatPromptTemplate.from_messages([
    ('system', tt_sys_prompt),
    ('system', tt_text_prompt),
    ('user', tt_user_prompt)
])


def make_title(state: SplitState):
    """
    结合已经划分的标题和上下文，为当前 block 分配一个标题
    :param state:
    :return:
    """

    tt_model = tt_prompt_template | llm_core.with_structured_output(Title)


    blocks = state['passage_blocks']
    current_idx = state['current_idx']
    print(current_idx)
    titles = state['titles']

    title3 = '标题:' + titles[current_idx - 3] if current_idx - 3 >= 0 else ''
    title2 = '标题:' + titles[current_idx - 2] if current_idx - 2 >= 0 else ''
    title1 = '标题:' + titles[current_idx - 1] if current_idx - 1 >= 0 else ''
    pre3 = '正文:' + blocks[current_idx - 3] if current_idx - 3 >= 0 else ''
    pre2 = '正文:' + blocks[current_idx - 2] if current_idx - 2 >= 0 else ''
    pre1 = '正文:' + blocks[current_idx - 1] if current_idx - 1 >= 0 else ''
    current = '需要你划分标题的正文' + blocks[current_idx]
    post1 = '正文:' + blocks[current_idx + 1] if current_idx + 1 < len(blocks) else ''
    post2 = '正文:' + blocks[current_idx + 2] if current_idx + 2 < len(blocks) else ''
    post3 = '正文:' + blocks[current_idx + 3] if current_idx + 3 < len(blocks) else ''

    try:
        res = tt_model.invoke({
            'title3': title3, 'title2': title2, 'title1': title1,
            'pre3': pre3, 'pre2': pre2, 'pre1': pre1,
            'current': current,
            'post1': post1, 'post2': post2, 'post3': post3
        })
        new_titles = titles
        new_titles.append(res.title_str)
    except BadRequestError as e:
        new_titles.append('--')

    print(new_titles[-1])

    return {
        'current_idx': 1,
        'titles': new_titles
    }

# openai.BadRequestError: Error code: 400 - {'error': {'code': 'data_inspection_failed', 'param': None, 'message': 'Input data may contain inappropriate content.', 'type': 'data_inspection_failed'}, 'id': 'chatcmpl-ffcbb12e-2263-93dc-b513-c646748bf41b'}
# openai.BadRequestError: Error code: 400 - {'error': {'code': 'invalid_parameter_error', 'param': None, 'message': 'Range of input length should be [1, 6000]', 'type': 'invalid_request_error'}, 'id': 'chatcmpl-30529382-c581-9475-9777-4609b85b2114'}


def route(state: SplitState):
    if state['current_idx'] < len(state['passage_blocks']):
        return 'make_title'
    else:
        return '__end__'


graph_builder = StateGraph(SplitState)
graph_builder.add_node('make_title', make_title)
graph_builder.add_edge(START, 'make_title')
graph_builder.add_conditional_edges('make_title', route, {'make_title': 'make_title', '__end__': '__end__'})
graph = graph_builder.compile()

if __name__ == '__main__':
    # from PIL import Image
    # from io import BytesIO
    # Image.open(BytesIO(graph.get_graph().draw_mermaid_png())).save('loop.png')

    with open('/Users/zhoushuzhe/Code/rain_agent/data/test.json', 'r', encoding='utf-8') as f:
        result = json.load(f)

    text_blocks = []
    for e_page in result:
        for e_block in e_page:
            if e_block['text']:
                text_blocks.append(e_block['text'])
            else:
                text_blocks.append(f"[{e_block['layout_type']}]")

    state = {
        'passage_blocks': text_blocks,
        'current_idx': 0,
        'titles': []
    }
    res = graph.invoke(state, config={'recursion_limit': 1000})

