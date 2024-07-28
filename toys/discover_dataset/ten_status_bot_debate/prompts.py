from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder

# sys = sys_base + sys_instruct

# in_history messages and not_in_history messages


sys_base = """
你是一个辩论专家，善于在复杂的辩论中说明自己的观点，指出别人的错误。
现在用户正在参加一场辩论会，会上总共有 10 个人。{person} 会告诉你辩论的内容，而你需要帮助 {person}思考如何发言

请你直接返回{person}所需的发言，不要额外的内容
"""

prompt_template = ChatPromptTemplate.from_messages(
    [
        ('system', sys_base),
        MessagesPlaceholder(variable_name='messages')
    ]
)

human_prompt = """
上一轮的历史发言为
{history_speech}

我是{person}, 我的立场是{stand}
请问我该如何发言？

1. 请直接告诉我发言内容，不要有多余的输出
2. 请不要长篇大论， 尽量精简你的发言
3. 请对和你立场不一致的发言给出反对的理由，然后提出你自己的观点，并给出解释
更多: {hint_with_status}

{person}:
"""