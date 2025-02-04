import getpass
import os

from langchain_core.messages import HumanMessage, AIMessage, ToolMessage

from my_agent.utils.model import llm_deepseek
from my_agent.utils.nodes import exists_action, call_deepseek, take_action
from my_agent.utils.state import AgentState
from my_agent.utils.tools import tools

from typing import TypedDict, Annotated
import operator
# from IPython.display import Image

from langchain_core.messages import AnyMessage, SystemMessage, HumanMessage
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
os.environ["LANGSMITH_TRACING"] = "true"
os.environ["LANGSMITH_API_KEY"] = os.getenv('LANGSMITH_API_KEY', default="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")

llm_with_tools = llm_deepseek.bind_tools(tools)
examples = [
    HumanMessage("317253 x 128472 + 4", name="example_user"),
    AIMessage(
        "",
        name="example_assistant",
        tool_calls=[
            {"name": "multiply", "args": {"a": 317253, "b": 128472}, "id": "1"}
        ],
    ),
    ToolMessage("40758127416", tool_call_id="1"),
    AIMessage(
        "",
        name="example_assistant",
        tool_calls=[{"name": "add", "args": {"a": "40758127416", "b": 4}, "id": "2"}],
    ),
    ToolMessage("40758127420", tool_call_id="2"),
    AIMessage(
        "317253 x 128472 + 4 = 40758127420",
        name="example_assistant",
    ),
    # 下一个例子

    HumanMessage("2 x 3 + 7", name="example_user"),
    AIMessage(
        "",
        name="example_assistant",
        tool_calls=[{"name": "multiply", "args": {"a": 2, "b": 3}, "id": "1"}],
    ),
    ToolMessage("6", tool_call_id="1"),
    AIMessage(
        "",
        name="example_assistant",
        tool_calls=[{"name": "add", "args": {"a": "6", "b": 7}, "id": "2"}],
    ),
    ToolMessage("13", tool_call_id="2"),
    AIMessage(
        "2 x 3 + 7 = 13",
        name="example_assistant",
    ),
    # 下一个例子
    HumanMessage("(6 + 7) x 9", name="example_user"),
    AIMessage(
        "",
        name="example_assistant",
        tool_calls=[{"name": "add", "args": {"a": 6, "b": 7}, "id": "1"}],
    ),
    ToolMessage("13", tool_call_id="1"),
    AIMessage(
        "",
        name="example_assistant",
        tool_calls=[{"name": "multiply", "args": {"a": 13, "b": 9}, "id": "2"}],
    ),
    ToolMessage("117", tool_call_id="2"),
    AIMessage(
        "(6 + 7) x 9 = 117",
        name="example_assistant",
    ),
]

class CalcuAgent:
    def __init__(self, model, tools, system="", checkpointer=None, examples=[]):
        self.system = system
        self.examples = examples
        graph = StateGraph(AgentState)
        graph.add_node("llm", self.to_call_deepseek)
        graph.set_entry_point("llm")
        graph.add_node("action", self.to_take_action)
        graph.add_conditional_edges(
            "llm", self.to_exists_action, {True: "action", False: END}
        )
        graph.add_edge("action", "llm")
        self.graph = graph.compile(checkpointer=checkpointer)

        self.tools = {t.name: t for t in tools}
        self.model = model.bind_tools(tools)

    def to_exists_action(self, state: AgentState):
        """

        :param state:
        :return:
        """

        return exists_action(state)

    def to_call_deepseek(self, state: AgentState):
        """

        :param state:
        :return:
        """
        return call_deepseek(state, [SystemMessage(self.system)] + self.examples + state["messages"], self.model, system, examples)

    def to_take_action(self, state: AgentState):
        """

        :param state:
        :return:
        """
        return take_action(state, self.tools)

    # def draw_graph(self):
    #     return Image(self.graph.get_graph().draw_png())
    #

memory = MemorySaver()

# 在系统提示词中，暗示大模型它的数据能力不行，要求它使用工具完成计算。不然，它就直接算出结果，本实验就没有意义了。
system = """你数学不好，但用计算器很在行。
以过去使用工具的情况为例，说明如何正确使用工具。
一次只能调用一个工具。
"""

agent = CalcuAgent(
    llm_deepseek, tools=tools, system=system, examples=examples, checkpointer=memory
).graph
#
# user_input = "10 x 20 + 200 + (14 + 16) x 10"
# config = {"configurable": {"thread_id": "1"}}
# events = agent.graph.stream(
#     {"messages": [HumanMessage(user_input)]}, config, stream_mode="values"
# )
#
#
# for event in events:
#     if "messages" in event:
#         event["messages"][-1].pretty_print()
#
#
