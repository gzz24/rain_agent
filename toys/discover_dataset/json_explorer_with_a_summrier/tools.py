import os

from langchain.tools import tool
from typing import List, Dict

from configs import config


def get_object_size():
    """get the element count in this object"""
    return 20


def get_object_type():
    """get the type of this object"""
    return 'dict'
