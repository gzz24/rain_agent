from langchain_core.prompts import ChatPromptTemplate

# Ref

ref_sys_prompt = """
你是一个论文阅读与调研助手，你将帮助用户在论文中筛选引用文献。

### instruction
1. 论文一般会在文献的结尾列举论文的引用文献（有些论文在引用文献后面还会有附录，注意区分）。在这些引用文献中，与论文的核心方法和背景直接相关的文献只占一部分，还有许多与核心方法不相关的、时间久远的论文。
2. 你需要将引用文献中，与本文的核心方法和背景相关的论文筛选出来，不超过15篇，不低于3篇。
3. 你需要给出引用文献的发表年份、作者、标题，然后详细解释该引用文献与论文内容的关联
4. 请按如下格式输出:
    (1) [年份] [作者] [文献名] 
        - [与本文的关联（详细）]
    (2) [年份] [作者] [文献名] 
        - [与本文的关联（详细）]
    ......
"""
ref_usr_prompt = """请你找出这篇文献的重要引用文献"""
ref_prompt_template = ChatPromptTemplate.from_messages(
    [
        ('system', ref_sys_prompt),
        ('system', '{passage}'),
        ('user', ref_usr_prompt)
    ]
)


extract_ref_sys_prompt = """
你是一个文本处理助手，擅长从文本中找出关键信息，以结构化形式输出
"""
extract_ref_user_prompt = """
请你从下面这段文本中，将引用文献的关键信息抽取出来，并返回给我
{ref}
"""
extract_ref_prompt_template = ChatPromptTemplate.from_messages(
    [
        ('system', extract_ref_sys_prompt),
        ('user', extract_ref_user_prompt)
    ]
)
extract_relocate_user_prompt = """
请你从下面这段文本中，将引用文献的引用位置、原文片段和引用原因抽取出来，并返回给我
{reason}
"""
extract_relocate_prompt_template = ChatPromptTemplate.from_messages(
    [
        ('system', extract_ref_sys_prompt),
        ('user', extract_relocate_user_prompt)
    ]
)


ref_relocate_sys_prompt = """
你是一个学术论文阅读的专家，你擅长从论文中找出与某篇引用文献相关的信息

### instruction
背景：一篇论文中通常会引用很多文献，每个引用都是对原文献工作的借鉴、改进或参考。
要求：对于给定的论文，用户会提供论文中的一篇引用文献，而你需要从论文中找出所有对该文献进行引用的位置，然后摘抄并总结。

你的返回格式应当如下：
    【引用位置1】
        章节位置：Introduction 2.3
        原文片段：...
        引用原因：论文借鉴了引用文献的xxx工作，对其中xxx方法进行了xxx改进...
    【引用位置2】
        章节位置：3.6 model structure
        原文片段：...
        引用原因：论文对文献的xxx结构进行了一些修改，比如...
    ......
    

"""
ref_relocate_user_prompt = """
这篇论文引用了下面这篇文献
年份:{year}，作者：{author}, 标题：{title}
引用文献与本文的关联是：{how_related}

请你仔细阅读这篇论文，帮我找出文中所有引用该文献的位置、原文片段，以及具体是如何用到了文献的工作。

"""
ref_relocate_prompt_template = ChatPromptTemplate.from_messages(
    [
        ('system', ref_relocate_sys_prompt),
        ('system', '{passage}'),
        ("user", ref_relocate_user_prompt)
    ]
)

# Control

control_sys_prompt = """
你是一个论文阅读助手，你将会帮助用户阅读一篇论文。
在阅读论文时，你首先需要对论文进行摘要总结。接下来，你需要对论文的实验章节进行提炼总结，并找出论文中的重要引用文献
"""
control_prompt_template = ChatPromptTemplate.from_messages(
    [
        ('system', control_sys_prompt),
        ('system', '{passage}'),
        ('user', '请帮我总结这篇论文')
    ]
)


# Intro

intro_sys_prompt = """
你是一个NLP领域的专家，你对前沿知识十分了解。
你非常擅长阅读论文，你能够从论文的介绍章节(Introduction)获取论文的主要工作以及相关背景。你将帮助用户完成对论文的Introduction章节的概括总结

### instruction
1. 论文的Introduction一般是对论文的总体介绍，通常包括了“背景知识”，“本文所讨论的主要研究方向”，“论文在该方向的工作”，“论文的核心贡献”这四部分内容，以及其他内容。
2. 你需要结合论文全文去理解论文的Introduction部分，并做出快速总结
3. 完成快速总结之后，你需要从Introduction中分别找出1中提到的四个部分的内容
4. 你需要按照如下格式返回你的总结结果：

[Introduction总体介绍]
...
[背景知识]
...
[本文内容涉及的研究方向]
...
[本文在该方向上的工作]
...
[本文的核心贡献]
...
[其他信息]
...

"""
intro_user_prompt = """请你对这篇文献的Introduction章节进行总结"""
intro_prompt_template = ChatPromptTemplate.from_messages(
    [
        ('system', intro_sys_prompt),
        ('system', '{passage}'),
        ('user', intro_user_prompt)
    ]
)


# Main Content (extract from intro)

intro_extract_user_prompt = """
请你从下面这篇论文的总结中，提炼出论文的主要贡献，然后返回给我

{summary}
"""
intro_extract_prompt_template = ChatPromptTemplate.from_messages(
    [
        ('system', extract_ref_sys_prompt),
        ('user', intro_extract_user_prompt)
    ]
)

main_sys_prompt = """
你是一个学术论文阅读的专家，擅长对论文的某个方面的内容进行提炼总结
"""
main_user_prompt = """
请你从提供的论文中，将其中与下面的内容有关的部分进行提炼与总结：

{point}

请总结与提炼：
"""
main_prompt_template = ChatPromptTemplate.from_messages([
    ('system', main_sys_prompt),
    ('system', '{passage}'),
    ('user', main_user_prompt)
])