from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder


# sys = sys_base + sys_instruct

# in_history messages and not_in_history messages


sys_base = """
你是一个知识广泛的助手
"""
sys_instruct = """
你将帮助探索数据集的分布特性
"""


sys_prompt = sys_base + sys_instruct


prompt_template = ChatPromptTemplate.from_messages(
    [
        ('system', sys_prompt),
        MessagesPlaceholder(variable_name='messages')
    ]
)


summary_instruct = """
你将对用户给出的这段对话进行总结，找出在对话中`assistant`角色所犯的错误，并用一句话总结该错误
"""
summary_prompt_template = ChatPromptTemplate.from_messages(
    [
        ('system', sys_base + summary_instruct),
        ('user', '{dialog}')
    ]
)