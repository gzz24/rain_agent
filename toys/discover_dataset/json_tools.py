# 固化代码
import os
import json

from langchain.tools import tool
from typing import List, Dict

from configs import config

with open(os.path.join(config['path']['dataset_root'], 'dev_tables.json'), 'r', encoding='utf-8') as f:
    json_obj = json.load(f)




# 固化模板

def get_json_type(**kwargs) -> str:
    """获得 json obj 的类型，一般是list, dict, str或int"""
    res = str(type(json_obj))

    print(f"get_json_type: {res}")
    return res


def get_json_length(**kwargs) -> str:
    """获得 json obj 的长度，只有 list 和 dict 才有长度"""
    if isinstance(json_obj, list) or isinstance(json_obj, dict):
        res = str(len(json_obj))
    else:
        res = "error:json_obj is not list or dict"
    print(f"get_json_length: {res}")
    return res


def get_json_list_element_type(**kwargs):
    """获取一个类型为 list 的 json obj 所包含 list 元素的类型,可能为list，dict，str 或 int"""
    try:
        res = str(type(json_obj[0]))
    except Exception as e:
        res = str(e)
    print(f"get_json_list_element_type: {res}")
    return res


def get_json_list_element_at_idx(idx: int, **kwargs):
    """获取一个类型为 list 的 json obj 的下标为`idx`的元素"""

    try:
        res = str(json_obj[idx])
        if len(res) > 300:
            res = f"因元素过大，部分信息被截断: {res[:300]}"
    except Exception as e:
        res = str(e)

    print(f"get_json_list_element_at_idx: {res}")
    return res


def get_json_dict_schema(**kwargs):
    """获取类型为 dict 的 json obj 的 key与 value 的类型"""

    try:
        key_type = str(type(list(json_obj.keys())[0]))
        value_type = str(type(list(json_obj.values())[0]))
        res = f'key type: {key_type}, value type: {value_type}'
    except Exception as e:
        res = str(e)

    print(f"get_json_dict_schema: {res}")
    return res