# test_multimodal.py
from src.agent import graph
from langchain_core.messages import HumanMessage

# Test with a table/chart question
config = {"configurable": {"thread_id": "multimodal-test"}}
response = graph.invoke(
    {"messages": [HumanMessage(content="describe figure 5 in the attention-is-all-you-need paper?")]},
    config=config
)

print(response["messages"][-1].content)