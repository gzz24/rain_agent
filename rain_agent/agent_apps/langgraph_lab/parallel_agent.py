import operator
import time
from typing import Annotated, Any

from typing_extensions import TypedDict

from langgraph.graph import StateGraph, START, END, add_messages


class State(TypedDict):
    # The operator.add reducer fn makes this append-only
    aggregate: Annotated[list, add_messages]


class ReturnNodeValue:
    def __init__(self, node_secret: str):
        self._value = node_secret

    def __call__(self, state: State) -> Any:
        print(f"Adding {self._value} to {state['aggregate']}")
        time.sleep(3)
        return {"aggregate": [self._value]}


builder = StateGraph(State)
builder.add_node("a", ReturnNodeValue("I'm A"))
builder.add_edge(START, "a")
builder.add_node("b", ReturnNodeValue("I'm B"))
builder.add_node("c", ReturnNodeValue("I'm C"))
builder.add_node("d", ReturnNodeValue("I'm D"))
builder.add_node("e", ReturnNodeValue("I'm E"))
builder.add_node("f", ReturnNodeValue("I'm F"))
builder.add_node("g", ReturnNodeValue("I'm G"))
builder.add_node("z", ReturnNodeValue("I'm Z"))
builder.add_edge("a", 'b')
builder.add_edge("a", 'c')
builder.add_edge("a", 'd')
builder.add_edge("a", 'e')
builder.add_edge("a", 'f')
builder.add_edge("a", 'g')
builder.add_edge(['b', 'c', 'd', 'e', 'f', 'g'], 'z')
builder.add_edge("z", END)
graph = builder.compile()


s = time.time()
graph.invoke({"aggregate": []}, {"configurable": {"thread_id": "foo"}})
s = time.time() - s
print(f"tool {s:.2f} seconds")


"""
Note
1. start_key可以是一个list，这样写方便点。end_key不能是list，开发者可能偷懒了
2. 可以并行运行，估计会有一个最大值，并且是可配置的
3. operator.add避免了重复
"""