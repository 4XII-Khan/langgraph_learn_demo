from langchain_core.messages import HumanMessage
from langgraph_sdk import get_sync_client

client = get_sync_client(url="http://127.0.0.1:2024")

for chunk in client.runs.stream(
    None,  # Threadless run
    "agent", # Name of assistant. Defined in langgraph.json.
    input={"messages": [HumanMessage("2 x 3 + 7")]},
    stream_mode="updates",
    config={"configurable": {"thread_id": "1"}}
):
    print(f"Receiving new event of type: {chunk.event} => {chunk.data}")
    print("\n")
