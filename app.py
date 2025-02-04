from langchain_core.messages import HumanMessage

from my_agent.agent import agent

user_input = "2 x 3 + 7"
config = {"configurable": {"thread_id": "1"}}

events = agent.stream(
    {"messages": [HumanMessage(user_input)]}, config, stream_mode="values"
)


for event in events:
    if "messages" in event:
        event["messages"][-1].pretty_print()


