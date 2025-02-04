from langchain_core.messages import SystemMessage, ToolMessage

from my_agent.utils.state import AgentState


def exists_action(state: AgentState):
    """

    :param state:
    :return:
    """
    result = state["messages"][-1]
    print(f"exists_action called with messages: {state['messages']}")
    return len(result.tool_calls) > 0


def call_deepseek(state: AgentState, messages, model, system, examples):
    """

    :param state:
    :return:
    """
    messages = [SystemMessage(system)] + examples + state["messages"]
    message = model.invoke(messages)
    print(f"call_deepseek called with messages: {messages}")
    return {"messages": [message]}


def take_action(state: AgentState, tools):
    """

    :param state:
    :return:
    """
    tool_calls = state["messages"][-1].tool_calls
    results = []
    for t in tool_calls:
        if not t["name"] in tools:  # check for bad tool name from LLM
            print("\n ....bad tool name....")
            result = "bad tool name, retry"  # instruct LLM to retry if bad
        else:
            print(f"Calling Action: {t['name']}")
            result = tools[t["name"]].invoke(t["args"])
            # print(f"action {t['name']}, result: {result}")
        results.append(
            ToolMessage(tool_call_id=t["id"], name=t["name"], content=str(result))
        )
    print("Back to the model!")
    return {"messages": results}

