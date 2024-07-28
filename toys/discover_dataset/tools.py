import os

from langchain.tools import tool
from typing import List, Dict

from configs import config


@tool
def get_dataset_files(dataset_name: str) -> List[str]:
    """get the files in the dataset named `dataset_name`"""
    res = os.listdir(config['path']['dataset_root'])
    print(f"tool: {res}")
    return res


@tool
def get_folder_files(folder_name: str) -> List[str]:
    """get the files in the folder of given name."""
    fname = os.path.join(config['path']['dataset_root'], folder_name)
    try:
        subdirs = os.listdir(fname)
        res = subdirs
        print(f"tool: {res}")
        return res
    except Exception as e:
        res = str(e)
        print(f"tool: {res}")
        return res

@tool
def get_file_basic_info(fname: str) -> Dict[str, str]:
    """get the basic info of the file"""
    res = {
        'file_name': {fname},
        'file_size': '25kb',
    }
    print(f"tool: {res}")
    return res


@tool
def get_json_info(json_fname: str) -> str:
    """get the detailed info of a json format file"""
    res = 'a list of dict'
    print(f"tool: {res}")
    return res


@tool
def get_json_part_content(json_fname: str) -> str:
    """get part of the content of a json format file"""
    fname = os.path.join(config['path']['dataset_root'], json_fname)

    with open(fname, 'r', encoding='utf-8') as f:
        fcontent = f.read().strip()
    res = fcontent[:200]
    print(f"tool: {res}")
    return res


@tool
def get_sql_info(sql_fname: str) -> str:
    """get the basic info of a sql file"""
    fname = os.path.join(config['path']['dataset_root'], sql_fname)

    with open(fname, 'r', encoding='utf-8') as f:
        fcontent = f.read().strip()
    res = fcontent[:200]
    print(f"tool: {res}")
    return res


@tool
def get_sql_part_content(sql_fname: str) -> str:
    """get part of the content of a sql format file"""
    res = """
select
    a,
    b,
    c,
from
    table
where
    d = 1
    """
    print(f"tool: {res}")
    return res

