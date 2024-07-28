from openai import OpenAI

functions = [{
    'name': 'get_dataset_files',
    'description': 'Get the files of the dataset',
    'parameters': {
        'type': 'object',
        'properties': {
            'dataset_name': {
                'type': 'string',
                'description': 'The name of the dataset',
            }
        },
        'required': ['dataset_name']
    },
}]

llm = OpenAI(
    api_key='sk-9c3717935de94e339b10435852fe760e',
    base_url='https://dashscope.aliyuncs.com/compatible-mode/v1'
)

res = llm.chat.completions.create(
    model='qwen-max',
    messages=[{
        'role': 'user',
        'content': '你好，请你帮我探索一下abddasd这个数据集'
    }, {
        'role': 'assistant',
        'content': '',
        'function_call': "dataset_name"
    }],
    functions=functions
)

print(res)